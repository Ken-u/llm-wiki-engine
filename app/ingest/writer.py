"""File writer: writes parsed FILE blocks to disk with merge support.

Ported from ingest.ts writeFileBlocks + page-merge.ts.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiofiles
import yaml

from app.ingest.parser import ParsedFileBlock
from app.llm import client as llm_client
from app.ingest.prompts import build_merge_prompt

logger = logging.getLogger(__name__)


def _parse_frontmatter(content: str) -> tuple[dict | None, str]:
    """Split a wiki page into (frontmatter_dict, body). Returns (None, content) if no frontmatter."""
    stripped = content.strip()
    if not stripped.startswith("---"):
        return None, content
    end = stripped.find("---", 3)
    if end == -1:
        return None, content
    fm_text = stripped[3:end].strip()
    body = stripped[end + 3:].strip()
    try:
        fm = yaml.safe_load(fm_text)
        if not isinstance(fm, dict):
            return None, content
        return fm, body
    except yaml.YAMLError:
        return None, content


def _merge_frontmatter(existing_fm: dict, incoming_fm: dict, source_filename: str) -> dict:
    """Merge frontmatter: union tags/related/sources, keep newer updated."""
    merged = {**existing_fm, **incoming_fm}

    for key in ("tags", "related", "sources"):
        existing_vals = set(existing_fm.get(key) or [])
        incoming_vals = set(incoming_fm.get(key) or [])
        merged[key] = sorted(existing_vals | incoming_vals)

    if source_filename:
        sources = set(merged.get("sources") or [])
        sources.add(source_filename)
        merged["sources"] = sorted(sources)

    return merged


def _rebuild_page(fm: dict, body: str) -> str:
    fm_str = yaml.dump(fm, allow_unicode=True, default_flow_style=False, sort_keys=False).strip()
    return f"---\n{fm_str}\n---\n\n{body}\n"


async def _llm_merge_bodies(existing_content: str, incoming_content: str, source_filename: str) -> str:
    """Use LLM to intelligently merge two versions of the same page."""
    user_msg = (
        f"## Existing version on disk\n\n{existing_content}\n\n---\n\n"
        f"## Incoming version (from {source_filename})\n\n{incoming_content}\n\n---\n\n"
        "Now produce the merged wiki page."
    )
    return await llm_client.stream_collect(
        build_merge_prompt(),
        user_msg,
        temperature=0.1,
        max_tokens=8192,
    )


async def write_file_blocks(
    project_dir: str,
    blocks: list[ParsedFileBlock],
    source_filename: str,
) -> list[str]:
    """Write parsed FILE blocks to disk. Merges with existing pages when they exist."""
    written: list[str] = []
    base = Path(project_dir)

    for block in blocks:
        target = base / block.path
        target.parent.mkdir(parents=True, exist_ok=True)

        # Special handling for log.md — always append
        if block.path == "wiki/log.md":
            existing = ""
            if target.exists():
                async with aiofiles.open(target, "r", encoding="utf-8") as f:
                    existing = await f.read()
            new_content = existing.rstrip() + "\n\n" + block.content.strip() + "\n"
            async with aiofiles.open(target, "w", encoding="utf-8") as f:
                await f.write(new_content)
            written.append(block.path)
            continue

        if target.exists():
            async with aiofiles.open(target, "r", encoding="utf-8") as f:
                existing_content = await f.read()

            if existing_content.strip():
                existing_fm, existing_body = _parse_frontmatter(existing_content)
                incoming_fm, incoming_body = _parse_frontmatter(block.content)

                if existing_fm and incoming_fm:
                    merged_fm = _merge_frontmatter(existing_fm, incoming_fm, source_filename)
                    # For index.md and overview.md, use incoming body directly (it's supposed to be the updated version)
                    if block.path in ("wiki/index.md", "wiki/overview.md"):
                        final_content = _rebuild_page(merged_fm, incoming_body)
                    else:
                        try:
                            merged_raw = await _llm_merge_bodies(existing_content, block.content, source_filename)
                            merged_fm2, merged_body = _parse_frontmatter(merged_raw)
                            if merged_fm2:
                                final_fm = _merge_frontmatter(existing_fm, merged_fm2, source_filename)
                                final_content = _rebuild_page(final_fm, merged_body)
                            else:
                                final_content = _rebuild_page(merged_fm, merged_raw)
                        except Exception:
                            logger.warning("LLM merge failed for %s, using incoming version", block.path)
                            final_content = _rebuild_page(merged_fm, incoming_body)
                else:
                    final_content = block.content
            else:
                final_content = block.content
        else:
            final_content = block.content

        async with aiofiles.open(target, "w", encoding="utf-8") as f:
            await f.write(final_content)
        written.append(block.path)

    return written
