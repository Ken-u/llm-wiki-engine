"""Hybrid search against the dedicated case index."""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.case_index.builder import CASE_INDEX_DIR, LANCEDB_TABLE, load_manifest
from app.case_index.models import CaseRecord

logger = logging.getLogger(__name__)

MAX_SNIPPET_CHARS = 200
RRF_K = 60


@dataclass
class MatchedSection:
    section: str
    snippet: str


@dataclass
class SearchResult:
    case_id: str
    title: str
    domain: str
    problem_summary: str
    root_cause: str
    resolution: str
    matched_sections: list[MatchedSection]
    score: float


def _case_index_path(project_dir: str) -> Path:
    return Path(project_dir) / CASE_INDEX_DIR


def _load_cases(project_dir: str) -> dict[str, CaseRecord]:
    cases_file = _case_index_path(project_dir) / "cases.jsonl"
    if not cases_file.exists():
        return {}
    records: dict[str, CaseRecord] = {}
    for line in cases_file.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        rec = CaseRecord.from_dict(d)
        records[rec.case_id] = rec
    return records


def _search_fts(project_dir: str, query: str, limit: int) -> list[tuple[str, str, float]]:
    """FTS5 keyword search. Returns list of (case_id, section, rank)."""
    db_path = _case_index_path(project_dir) / "keyword.sqlite"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT case_id, section, rank FROM case_fts WHERE case_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (query, limit * 3),
        ).fetchall()
        return [(r[0], r[1], float(r[2])) for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


async def _get_embeddings(texts: list[str]) -> list[list[float]]:
    from app.embedding.service import _get_embeddings as get_embeddings

    return await get_embeddings(texts)


async def _search_vector(
    project_dir: str, query: str, limit: int
) -> list[tuple[str, str, str, float]]:
    """LanceDB vector search. Returns (case_id, section, chunk_text, distance)."""
    lance_dir = str(_case_index_path(project_dir) / "lancedb")
    if not Path(lance_dir).exists():
        return []

    import lancedb

    embeddings = await _get_embeddings([query])
    if not embeddings:
        return []

    db = await lancedb.connect_async(lance_dir)
    table_names = await db.table_names()
    if LANCEDB_TABLE not in table_names:
        return []

    table = await db.open_table(LANCEDB_TABLE)
    try:
        results = await table.vector_search(embeddings[0]).limit(limit * 3).to_pandas()
    except ValueError:
        return []

    output = []
    for _, row in results.iterrows():
        output.append((
            row.get("case_id", ""),
            row.get("section", ""),
            row.get("chunk_text", ""),
            float(row.get("_distance", 0)),
        ))
    return output


async def search_cases(
    project_dir: str, query: str, *, limit: int = 3
) -> list[SearchResult]:
    """Hybrid FTS5 + vector search, RRF fusion, case-level aggregation."""
    manifest = load_manifest(project_dir)
    if manifest is None or not manifest.is_ready:
        return []

    limit = min(limit, 20)

    fts_hits = _search_fts(project_dir, query, limit)
    vec_hits = await _search_vector(project_dir, query, limit)

    # RRF fusion at case level
    case_scores: dict[str, float] = {}
    case_sections: dict[str, list[MatchedSection]] = {}

    for rank, (case_id, section, _rank) in enumerate(fts_hits):
        rrf = 1.0 / (RRF_K + rank + 1)
        case_scores[case_id] = case_scores.get(case_id, 0) + rrf
        case_sections.setdefault(case_id, [])

    for rank, (case_id, section, chunk_text, _dist) in enumerate(vec_hits):
        rrf = 1.0 / (RRF_K + rank + 1)
        case_scores[case_id] = case_scores.get(case_id, 0) + rrf
        secs = case_sections.setdefault(case_id, [])
        if not any(s.section == section for s in secs):
            secs.append(MatchedSection(
                section=section,
                snippet=chunk_text[:MAX_SNIPPET_CHARS],
            ))

    if not case_scores:
        return []

    ranked = sorted(case_scores.items(), key=lambda x: -x[1])[:limit]
    cases = _load_cases(project_dir)

    results = []
    for case_id, score in ranked:
        rec = cases.get(case_id)
        if rec is None:
            continue
        results.append(SearchResult(
            case_id=rec.case_id,
            title=rec.title,
            domain=rec.domain,
            problem_summary=rec.problem_summary[:MAX_SNIPPET_CHARS],
            root_cause=rec.root_cause[:MAX_SNIPPET_CHARS],
            resolution=rec.resolution[:MAX_SNIPPET_CHARS],
            matched_sections=case_sections.get(case_id, [])[:3],
            score=round(score, 4),
        ))

    return results


def read_case(
    project_dir: str, case_id: str, *, section: str | None = None
) -> dict | None:
    """Read a case record for agent consumption, optionally filtered to a section."""
    from app.case_index.parser import (
        _split_sections,
        match_section_query,
        sections_to_agent_dict,
        normalize_heading,
    )
    from app.wiki.frontmatter import parse_frontmatter

    cases = _load_cases(project_dir)
    rec = cases.get(case_id)
    if rec is None:
        return None

    source = Path(project_dir) / rec.source_path
    if not source.exists():
        return {
            "source_type": "ticket_case_index",
            "case_id": case_id,
            "error": f"Case source not available for case_id: {case_id}",
        }

    text = source.read_text(encoding="utf-8", errors="replace")
    _, body = parse_frontmatter(text)
    parsed_sections = _split_sections(body)
    available, sections_dict = sections_to_agent_dict(parsed_sections)

    if section is not None:
        matched = match_section_query(section, parsed_sections)
        if matched is None:
            return {
                "source_type": "ticket_case_index",
                "case_id": case_id,
                "title": rec.title,
                "error": f"Section not found: {section}",
                "available_sections": available,
            }
        heading, _field_name, content = matched
        return {
            "source_type": "ticket_case_index",
            "case_id": case_id,
            "title": rec.title,
            "section": normalize_heading(heading),
            "content": content[:6000],
        }

    return {
        "source_type": "ticket_case_index",
        "case_id": rec.case_id,
        "title": rec.title,
        "domain": rec.domain,
        "tags": rec.tags,
        "available_sections": available,
        "sections": sections_dict,
    }


def _find_case_source_path(project_dir: str, case_id: str) -> Path | None:
    cases = _load_cases(project_dir)
    rec = cases.get(case_id)
    if rec is not None:
        candidate = Path(project_dir) / rec.source_path
        if candidate.exists():
            return candidate

    src = Path(project_dir) / "raw" / "sources"
    if not src.exists():
        return None
    for path in src.rglob("*.md"):
        if path.is_file() and path.stem == case_id:
            return path
    return None


def read_case_source(project_dir: str, case_id: str) -> dict | None:
    """Read case raw markdown source for UI preview."""
    source_path = _find_case_source_path(project_dir, case_id)
    if source_path is None:
        return None

    raw_content = source_path.read_text(encoding="utf-8", errors="replace")
    detail = read_case(project_dir, case_id)
    title = detail.get("title", case_id) if detail else case_id

    return {
        "source_type": "ticket_case_index",
        "case_id": case_id,
        "title": title,
        "raw_content": raw_content,
    }
