"""Fast-path knowledge lookup: definition/concept retrieval without LLM.

Searches wiki/entities/ and wiki/concepts/ for matching pages,
then extracts the definition/overview section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.search.bm25 import search_bm25
from app.wiki.frontmatter import parse_frontmatter


@dataclass
class FastLookupResult:
    path: str
    title: str
    content: str
    matched_by: str  # "slug" | "title" | "bm25"


_DEFINITION_HEADINGS = re.compile(
    r"^##\s*(定义|概述|简介|概念|Definition|Overview|Summary|Introduction)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_CONCEPT_DIRS = ("wiki/entities", "wiki/concepts")

MAX_EXCERPT_CHARS = 2000
BM25_SCORE_THRESHOLD = 2.0


def _slugify(term: str) -> str:
    """Convert a term to kebab-case slug for filename matching."""
    s = term.strip().lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", s)
    return s.strip("-")


def _extract_definition_section(body: str) -> str:
    """Extract the definition/overview section from page body."""
    m = _DEFINITION_HEADINGS.search(body)
    if m:
        start = m.end()
        next_heading = re.search(r"^##\s", body[start:], re.MULTILINE)
        end = start + next_heading.start() if next_heading else len(body)
        excerpt = body[start:end].strip()
        if excerpt:
            return excerpt[:MAX_EXCERPT_CHARS]

    # Fallback: take content from first heading to next ## heading
    first_h = re.search(r"^#\s+.+$", body, re.MULTILINE)
    if first_h:
        after_h = first_h.end()
        next_h2 = re.search(r"^##\s", body[after_h:], re.MULTILINE)
        end = after_h + next_h2.start() if next_h2 else min(after_h + MAX_EXCERPT_CHARS, len(body))
        excerpt = body[after_h:end].strip()
        if excerpt:
            return excerpt[:MAX_EXCERPT_CHARS]

    # Last resort: first N chars of body
    return body[:MAX_EXCERPT_CHARS].strip()


def _try_slug_match(project_dir: str, term: str) -> FastLookupResult | None:
    """Try to find a page by slug (kebab-case filename match)."""
    slug = _slugify(term)
    if not slug:
        return None

    base = Path(project_dir)
    for subdir in _CONCEPT_DIRS:
        candidate = base / subdir / f"{slug}.md"
        if candidate.exists():
            content = candidate.read_text(encoding="utf-8", errors="replace")
            meta, body = parse_frontmatter(content)
            rel_path = str(candidate.relative_to(base))
            return FastLookupResult(
                path=rel_path,
                title=meta.title or slug,
                content=_extract_definition_section(body),
                matched_by="slug",
            )
    return None


def _try_title_match(project_dir: str, term: str) -> FastLookupResult | None:
    """Scan entity/concept pages for exact title match."""
    term_lower = term.strip().lower()
    base = Path(project_dir)

    for subdir in _CONCEPT_DIRS:
        dir_path = base / subdir
        if not dir_path.exists():
            continue
        for md_file in dir_path.glob("*.md"):
            content = md_file.read_text(encoding="utf-8", errors="replace")
            meta, body = parse_frontmatter(content)
            if meta.title.lower() == term_lower:
                rel_path = str(md_file.relative_to(base))
                return FastLookupResult(
                    path=rel_path,
                    title=meta.title,
                    content=_extract_definition_section(body),
                    matched_by="title",
                )
    return None


def _try_bm25_match(project_dir: str, term: str) -> FastLookupResult | None:
    """BM25 keyword search filtered to entity/concept pages."""
    results = search_bm25(project_dir, term, top_k=5)
    for r in results:
        if r.score < BM25_SCORE_THRESHOLD:
            break
        if any(r.path.startswith(d) for d in _CONCEPT_DIRS):
            full_path = Path(project_dir) / r.path
            if full_path.exists():
                content = full_path.read_text(encoding="utf-8", errors="replace")
                _, body = parse_frontmatter(content)
                return FastLookupResult(
                    path=r.path,
                    title=r.title,
                    content=_extract_definition_section(body),
                    matched_by="bm25",
                )
    return None


def fast_lookup(project_dir: str, term: str) -> FastLookupResult | None:
    """Attempt fast definition lookup without LLM.

    Tries in order: slug match, title match, BM25 filtered to concepts/entities.
    Returns None if no suitable page is found.
    """
    result = _try_slug_match(project_dir, term)
    if result:
        return result

    result = _try_title_match(project_dir, term)
    if result:
        return result

    return _try_bm25_match(project_dir, term)


_COMPLEX_QUERY_MARKERS = re.compile(
    r"(差异|区别|不同|对比|比较|关系|如何|怎么|为什么|什么时候|哪些|能否|是否|"
    r"how|why|when|which|compare|difference|versus|vs\.?)",
    re.IGNORECASE,
)


def is_definition_query(message: str) -> bool:
    """Heuristic: detect if a user message is a simple definition/concept lookup.

    Returns False for comparison questions, how-to questions, etc.
    """
    msg = message.strip()

    # Reject if contains complex query markers (comparison, how-to, etc.)
    if _COMPLEX_QUERY_MARKERS.search(msg):
        return False

    # Very short messages that are just a term (<=8 chars, no spaces for CJK)
    if len(msg) <= 8 and " " not in msg:
        return True

    # Pattern matching for explicit definition questions
    definition_patterns = [
        r"^什么是\S+$",
        r"^\S+是什么[？?]?$",
        r"^\S+的定义[？?]?$",
        r"^\S+的概念[？?]?$",
        r"^(define|what is)\s+\S+$",
    ]
    for pat in definition_patterns:
        if re.match(pat, msg, re.IGNORECASE):
            return True

    # Single word/token (ASCII term like "GMS", "EDLA")
    word_count = len(re.findall(r"\S+", msg))
    if word_count == 1:
        return True

    return False


def extract_term_from_definition_query(message: str) -> str:
    """Extract the term from a definition-style query."""
    msg = message.strip().rstrip("？?。.")

    patterns = [
        (r"^什么是(.+)$", 1),
        (r"^(.+)是什么$", 1),
        (r"^(.+)的定义$", 1),
        (r"^(.+)的概念$", 1),
        (r"^(?:define|what is)\s+(.+)$", 1),
    ]
    for pat, group in patterns:
        m = re.match(pat, msg, re.IGNORECASE)
        if m:
            return m.group(group).strip()

    return msg
