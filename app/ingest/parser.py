"""FILE / REVIEW block parser for LLM generation output.

Ported from the line-level state machine in ingest.ts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

OPENER_LINE = re.compile(r"^---\s*FILE:\s*(.+?)\s*---\s*$", re.IGNORECASE)
CLOSER_LINE = re.compile(r"^---\s*END\s+FILE\s*---\s*$", re.IGNORECASE)
REVIEW_OPENER = re.compile(r"^---\s*REVIEW:\s*(.+?)\s*---\s*$", re.IGNORECASE)
REVIEW_CLOSER = re.compile(r"^---\s*END\s+REVIEW\s*---\s*$", re.IGNORECASE)


@dataclass
class ParsedFileBlock:
    path: str
    content: str


@dataclass
class ReviewBlock:
    review_type: str
    title: str
    description: str
    options: list[str] = field(default_factory=list)
    pages: list[str] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)


@dataclass
class ParseResult:
    blocks: list[ParsedFileBlock]
    reviews: list[ReviewBlock]
    warnings: list[str]


def is_safe_ingest_path(p: str) -> bool:
    if not p or not p.strip():
        return False
    if re.search(r"[\x00-\x1f]", p):
        return False
    if p.startswith("/") or p.startswith("\\"):
        return False
    if re.match(r"^[A-Za-z]:", p):
        return False
    segments = re.split(r"[/\\]", p)
    for seg in segments:
        if seg == "..":
            return False
        if not seg:
            continue
        if seg.endswith(" ") or seg.endswith("."):
            return False
    if not p.startswith("wiki/"):
        return False
    return True


def parse_file_blocks(text: str) -> ParseResult:
    """Parse ---FILE: path--- ... ---END FILE--- blocks from LLM output."""
    blocks: list[ParsedFileBlock] = []
    reviews: list[ReviewBlock] = []
    warnings: list[str] = []

    lines = text.split("\n")
    current_path: str | None = None
    current_lines: list[str] = []
    in_review = False
    review_header: str = ""
    review_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Check for FILE opener
        m = OPENER_LINE.match(stripped)
        if m:
            if current_path is not None:
                warnings.append(f"Unclosed FILE block for {current_path}, auto-closing")
                content = "\n".join(current_lines).strip()
                if content:
                    blocks.append(ParsedFileBlock(path=current_path, content=content))
            path = m.group(1).strip()
            if is_safe_ingest_path(path):
                current_path = path
                current_lines = []
            else:
                warnings.append(f"Rejected unsafe path: {path}")
                current_path = None
                current_lines = []
            continue

        # Check for FILE closer
        if CLOSER_LINE.match(stripped):
            if current_path is not None:
                content = "\n".join(current_lines).strip()
                if content:
                    blocks.append(ParsedFileBlock(path=current_path, content=content))
                current_path = None
                current_lines = []
            continue

        # Check for REVIEW opener
        rm = REVIEW_OPENER.match(stripped)
        if rm and current_path is None:
            in_review = True
            review_header = rm.group(1).strip()
            review_lines = []
            continue

        # Check for REVIEW closer
        if REVIEW_CLOSER.match(stripped) and in_review:
            parts = review_header.split("|", 1)
            rtype = parts[0].strip() if parts else "suggestion"
            rtitle = parts[1].strip() if len(parts) > 1 else rtype
            review = ReviewBlock(review_type=rtype, title=rtitle, description="")
            desc_lines = []
            for rl in review_lines:
                if rl.startswith("OPTIONS:"):
                    review.options = [o.strip() for o in rl[8:].split("|")]
                elif rl.startswith("PAGES:"):
                    review.pages = [p.strip() for p in rl[6:].split(",")]
                elif rl.startswith("SEARCH:"):
                    review.search_queries = [q.strip() for q in rl[7:].split("|")]
                else:
                    desc_lines.append(rl)
            review.description = "\n".join(desc_lines).strip()
            reviews.append(review)
            in_review = False
            continue

        if in_review:
            review_lines.append(line)
        elif current_path is not None:
            current_lines.append(line)

    # Handle unclosed trailing block
    if current_path is not None:
        warnings.append(f"Unclosed FILE block for {current_path} at end of output, auto-closing")
        content = "\n".join(current_lines).strip()
        if content:
            blocks.append(ParsedFileBlock(path=current_path, content=content))

    return ParseResult(blocks=blocks, reviews=reviews, warnings=warnings)
