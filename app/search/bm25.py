"""BM25 keyword search over wiki pages.

Reads all wiki markdown files from disk and runs BM25 ranking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    path: str
    page_id: str
    title: str
    score: float
    snippet: str


def _project_relative_path(path: Path, project_dir: Path) -> str:
    return path.relative_to(project_dir).as_posix()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _extract_title(content: str) -> str:
    """Extract title from frontmatter or first heading."""
    for line in content.split("\n"):
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
        m = re.match(r"^#+\s+(.+)", line)
        if m:
            return m.group(1).strip()
    return ""


def _snippet(content: str, query_tokens: list[str], max_len: int = 200) -> str:
    """Extract a relevant snippet containing query terms."""
    content_lower = content.lower()
    best_pos = 0
    best_score = 0
    for i in range(0, len(content) - 50, 50):
        window = content_lower[i:i + max_len]
        score = sum(1 for t in query_tokens if t in window)
        if score > best_score:
            best_score = score
            best_pos = i
    return content[best_pos:best_pos + max_len].strip()


def search_bm25(project_dir: str, query: str, top_k: int = 10) -> list[BM25Result]:
    wiki_dir = Path(project_dir) / "wiki"
    if not wiki_dir.exists():
        return []

    pages: list[tuple[str, str, str]] = []  # (rel_path, content, title)
    for md_file in wiki_dir.rglob("*.md"):
        rel = _project_relative_path(md_file, Path(project_dir))
        content = md_file.read_text(encoding="utf-8", errors="replace")
        title = _extract_title(content)
        pages.append((rel, content, title))

    if not pages:
        return []

    corpus = [_tokenize(content) for _, content, _ in pages]
    bm25 = BM25Okapi(corpus)
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]

    results = []
    for idx, score in ranked:
        if score <= 0:
            continue
        rel_path, content, title = pages[idx]
        page_id = Path(rel_path).stem
        results.append(BM25Result(
            path=rel_path,
            page_id=page_id,
            title=title,
            score=float(score),
            snippet=_snippet(content, query_tokens),
        ))

    return results
