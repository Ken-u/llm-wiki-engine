"""Wiki page path resolution helpers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.search.vector import VectorResult, search_vector

logger = logging.getLogger(__name__)


@dataclass
class WikiPageResolution:
    path: Path
    info: dict


def wiki_fallback_query(path: str) -> str:
    stem = Path(path).stem or path
    return re.sub(r"[_\-/\\]+", " ", stem).strip()


def normalize_page_id(value: str) -> str:
    stem = Path(value).stem
    return re.sub(r"[^a-z0-9_-]", "-", stem.lower())


def _safe_join(base: str, rel_path: str) -> Path | None:
    if ".." in rel_path or rel_path.startswith("/") or rel_path.startswith("\\"):
        return None
    root = Path(base).resolve()
    candidate = (root / rel_path).resolve()
    if root != candidate and root not in candidate.parents:
        return None
    return candidate


def resolve_vector_result_path(project_dir: str, result: VectorResult) -> Path | None:
    direct = _safe_join(project_dir, result.path)
    if direct is not None and direct.exists() and direct.is_file():
        return direct

    wiki_dir = Path(project_dir) / "wiki"
    if not wiki_dir.exists():
        return None

    target_ids = {
        normalize_page_id(result.page_id),
        normalize_page_id(result.path),
    }
    for candidate in wiki_dir.rglob("*.md"):
        if candidate.name.startswith("."):
            continue
        if normalize_page_id(candidate.name) in target_ids:
            return candidate
    return None


async def resolve_missing_wiki_page(
    project_dir: str,
    requested_path: str,
    distance_threshold: float,
) -> WikiPageResolution | None:
    query = wiki_fallback_query(requested_path)
    if not query:
        return None

    try:
        results = await search_vector(project_dir, query, top_k=5)
    except Exception:
        logger.exception("Wiki vector fallback failed for missing page: %s", requested_path)
        return None

    eligible = [result for result in results if result.score <= distance_threshold]
    for result in sorted(eligible, key=lambda item: item.score):
        resolved_path = resolve_vector_result_path(project_dir, result)
        if resolved_path is None:
            continue
        return WikiPageResolution(
            path=resolved_path,
            info={
                "method": "vector",
                "requested_path": requested_path,
                "query": query,
                "matched_path": resolved_path.relative_to(Path(project_dir)).as_posix(),
                "matched_page_id": result.page_id,
                "distance": result.score,
                "threshold": distance_threshold,
            },
        )
    return None
