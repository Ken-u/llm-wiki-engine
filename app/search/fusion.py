"""RRF (Reciprocal Rank Fusion) for combining BM25 + vector results.

Ported from src-tauri/src/commands/search.rs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import get_config
from app.search.bm25 import BM25Result
from app.search.vector import VectorResult


@dataclass
class FusedResult:
    path: str
    page_id: str
    title: str
    score: float
    snippet: str
    sources: list[str]  # which search modes contributed


def rrf_fusion(
    keyword_results: list[BM25Result],
    vector_results: list[VectorResult],
    query: str,
) -> list[FusedResult]:
    """Combine keyword and vector results using RRF with bonus scoring."""
    cfg = get_config().search
    k = cfg.rrf_k

    scores: dict[str, float] = {}
    titles: dict[str, str] = {}
    snippets: dict[str, str] = {}
    sources: dict[str, set] = {}

    query_lower = query.lower()
    query_tokens = re.findall(r"\w+", query_lower)

    for rank, r in enumerate(keyword_results):
        rrf = 1.0 / (k + rank + 1)
        scores[r.path] = scores.get(r.path, 0) + rrf
        titles[r.path] = r.title
        snippets[r.path] = r.snippet
        sources.setdefault(r.path, set()).add("keyword")

    for rank, r in enumerate(vector_results):
        rrf = 1.0 / (k + rank + 1)
        scores[r.path] = scores.get(r.path, 0) + rrf
        if r.path not in titles:
            titles[r.path] = r.page_id
        if r.path not in snippets:
            snippets[r.path] = r.chunk_text[:200]
        sources.setdefault(r.path, set()).add("vector")

    # Bonus scoring
    for path, score in list(scores.items()):
        page_id = re.sub(r"^wiki/(.+)\.md$", r"\1", path).split("/")[-1]

        # Filename exact match bonus
        if page_id.lower() == query_lower or page_id.lower().replace("-", " ") == query_lower:
            scores[path] += cfg.filename_exact_bonus

        # Title phrase match bonus
        title = titles.get(path, "").lower()
        if query_lower in title:
            scores[path] += cfg.phrase_in_title_bonus

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    return [
        FusedResult(
            path=path,
            page_id=re.sub(r"^wiki/(.+)\.md$", r"\1", path).split("/")[-1],
            title=titles.get(path, ""),
            score=score,
            snippet=snippets.get(path, ""),
            sources=sorted(sources.get(path, set())),
        )
        for path, score in ranked
    ]
