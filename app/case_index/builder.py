"""Full-rebuild pipeline for the dedicated case index."""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import lancedb
import pyarrow as pa

from app.case_index.models import CaseManifest, CaseRecord
from app.case_index.parser import parse_case_markdown
from app.config import get_config
from app.embedding.service import _get_embeddings

logger = logging.getLogger(__name__)

CASE_INDEX_DIR = ".llm-wiki/case-index"
CASE_INDEX_TMP_DIR = ".llm-wiki/case-index.tmp"
LANCEDB_TABLE = "case_chunks_v1"
EMBEDDING_BATCH_SIZE = 64


def _case_index_path(project_dir: str) -> Path:
    return Path(project_dir) / CASE_INDEX_DIR


def load_manifest(project_dir: str) -> CaseManifest | None:
    mf = _case_index_path(project_dir) / "manifest.json"
    if not mf.exists():
        return None
    try:
        return CaseManifest.from_dict(json.loads(mf.read_text(encoding="utf-8")))
    except Exception:
        return None


STALE_MARKER = ".llm-wiki/case-index-stale"


def mark_case_index_stale(project_dir: str) -> None:
    marker = Path(project_dir) / STALE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        datetime.now(timezone.utc).isoformat(), encoding="utf-8"
    )


def clear_case_index_stale(project_dir: str) -> None:
    marker = Path(project_dir) / STALE_MARKER
    if marker.exists():
        marker.unlink()


def is_case_index_stale(project_dir: str) -> bool:
    return (Path(project_dir) / STALE_MARKER).exists()


def _scan_sources(project_dir: str) -> list[Path]:
    src = Path(project_dir) / "raw" / "sources"
    if not src.exists():
        return []
    return sorted(
        p for p in src.rglob("*.md")
        if p.is_file() and not p.name.startswith(".")
    )


def _build_fts(db_path: Path, records: list[CaseRecord], chunks: list[dict]):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS case_fts USING fts5("
        "case_id, title, section, content, tags, domain, "
        "tokenize='unicode61'"
        ")"
    )
    conn.execute("DELETE FROM case_fts")

    for rec in records:
        conn.execute(
            "INSERT INTO case_fts (case_id, title, section, content, tags, domain) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                rec.case_id,
                rec.title,
                "",
                " ".join(filter(None, [
                    rec.problem_summary,
                    rec.root_cause,
                    rec.resolution,
                    rec.diagnosis_steps,
                ])),
                " ".join(rec.tags),
                rec.domain,
            ),
        )

    for c in chunks:
        conn.execute(
            "INSERT INTO case_fts (case_id, title, section, content, tags, domain) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (c["case_id"], "", c["section"], c["chunk_text"], "", ""),
        )

    conn.commit()
    conn.close()


def _lance_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id", pa.string()),
        pa.field("case_id", pa.string()),
        pa.field("source_path", pa.string()),
        pa.field("section", pa.string()),
        pa.field("heading_path", pa.string()),
        pa.field("chunk_text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
    ])


def _write_manifest(path: Path, manifest: CaseManifest):
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def rebuild_case_index(project_dir: str) -> CaseManifest:
    """Scan raw/sources/*.md, build FTS5 + LanceDB case index, write manifest."""
    cfg = get_config()
    base = Path(project_dir)
    tmp_dir = base / CASE_INDEX_TMP_DIR
    final_dir = base / CASE_INDEX_DIR

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    sources = _scan_sources(project_dir)
    errors: list[str] = []
    all_records: list[CaseRecord] = []
    all_chunks: list[dict] = []

    for src in sources:
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
            rel_path = str(src.relative_to(base))
            record, chunks = parse_case_markdown(text, rel_path)
            all_records.append(record)
            for c in chunks:
                all_chunks.append({
                    "chunk_id": c.chunk_id,
                    "case_id": c.case_id,
                    "source_path": c.source_path,
                    "section": c.section,
                    "heading_path": c.heading_path,
                    "chunk_text": c.chunk_text,
                })
        except Exception as exc:
            errors.append(f"{src.name}: {exc}")
            logger.warning("Failed to parse case %s: %s", src.name, exc)

    if not all_chunks:
        manifest = CaseManifest(
            status="failed",
            built_at=datetime.now(timezone.utc).isoformat(),
            source_count=len(sources),
            case_count=0,
            chunk_count=0,
            embedding_model=cfg.embedding.model,
            embedding_dimensions=cfg.embedding.dimensions or 0,
            errors=errors or ["No chunks produced from source files"],
            index_version=1,
        )
        shutil.rmtree(tmp_dir)
        return manifest

    # Write cases.jsonl
    with open(tmp_dir / "cases.jsonl", "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    # Build FTS5
    _build_fts(tmp_dir / "keyword.sqlite", all_records, all_chunks)

    # Embed chunks in batches
    try:
        texts = [c["chunk_text"] for c in all_chunks]
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            vecs = await _get_embeddings(batch)
            all_vectors.extend(vecs)

        if len(all_vectors) != len(all_chunks):
            raise RuntimeError(
                f"Vector count mismatch: {len(all_chunks)} chunks, {len(all_vectors)} vectors"
            )
    except Exception as exc:
        logger.error("Embedding failed during case index build: %s", exc)
        manifest = CaseManifest(
            status="failed",
            built_at=datetime.now(timezone.utc).isoformat(),
            source_count=len(sources),
            case_count=len(all_records),
            chunk_count=0,
            embedding_model=cfg.embedding.model,
            embedding_dimensions=cfg.embedding.dimensions or 0,
            errors=[f"Embedding failed: {exc}"],
            index_version=1,
        )
        shutil.rmtree(tmp_dir)
        return manifest

    # Write LanceDB
    vector_dim = len(all_vectors[0]) if all_vectors else 0
    lance_dir = str(tmp_dir / "lancedb")
    db = await lancedb.connect_async(lance_dir)
    records_with_vec = []
    for chunk, vec in zip(all_chunks, all_vectors):
        records_with_vec.append({**chunk, "vector": vec})
    await db.create_table(
        LANCEDB_TABLE, records_with_vec, schema=_lance_schema(vector_dim)
    )

    # Write manifest
    manifest = CaseManifest(
        status="ready",
        built_at=datetime.now(timezone.utc).isoformat(),
        source_count=len(sources),
        case_count=len(all_records),
        chunk_count=len(all_chunks),
        embedding_model=cfg.embedding.model,
        embedding_dimensions=vector_dim,
        errors=errors,
        index_version=1,
    )
    _write_manifest(tmp_dir / "manifest.json", manifest)

    # Atomic swap
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    clear_case_index_stale(project_dir)

    logger.info(
        "Case index rebuilt: %d cases, %d chunks from %d sources",
        manifest.case_count,
        manifest.chunk_count,
        manifest.source_count,
    )
    return manifest
