"""Markdown-aware recursive text chunker.

Ported from text-chunker.ts — splits markdown documents into embeddable
chunks while preserving heading context and respecting code blocks / tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_CHUNK_SIZE = 512
DEFAULT_OVERLAP = 64
MIN_CHUNK_SIZE = 64

# Split priorities: try headers first, then paragraphs, then sentences, then words
_SPLIT_PATTERNS = [
    re.compile(r"\n#{1,6}\s"),     # Markdown headings
    re.compile(r"\n\n"),            # Double newline (paragraphs)
    re.compile(r"\n"),              # Single newline
    re.compile(r"[.!?]\s"),        # Sentence boundaries
    re.compile(r"\s"),              # Word boundaries
]


@dataclass
class Chunk:
    text: str
    heading_path: str
    chunk_index: int


def _extract_heading_path(text: str) -> str:
    """Build a heading path like "## Section > ### Subsection" from headings in the text."""
    headings: list[str] = []
    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.+)", line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop headings that are at the same or deeper level
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, f"{'#' * level} {title}"))
    return " > ".join(h[1] for h in headings) if headings else ""


def _find_split_point(text: str, target: int) -> int:
    """Find the best split point near target position."""
    best = target
    for pattern in _SPLIT_PATTERNS:
        # Search backward from target
        for m in pattern.finditer(text[:target + 100]):
            pos = m.start()
            if abs(pos - target) < abs(best - target) and pos > MIN_CHUNK_SIZE:
                best = pos
                break
        if best != target:
            break
    return max(best, MIN_CHUNK_SIZE)


def chunk_markdown(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split markdown text into overlapping chunks preserving heading context."""
    if not text or not text.strip():
        return []

    if len(text) <= chunk_size:
        return [Chunk(text=text.strip(), heading_path=_extract_heading_path(text), chunk_index=0)]

    chunks: list[Chunk] = []
    pos = 0
    idx = 0

    while pos < len(text):
        end = min(pos + chunk_size, len(text))

        if end < len(text):
            split_at = _find_split_point(text[pos:end], chunk_size)
            end = pos + split_at

        chunk_text = text[pos:end].strip()
        if chunk_text:
            heading_path = _extract_heading_path(text[:end])
            chunks.append(Chunk(text=chunk_text, heading_path=heading_path, chunk_index=idx))
            idx += 1

        pos = max(end - overlap, pos + MIN_CHUNK_SIZE)
        if pos >= len(text):
            break

    return chunks
