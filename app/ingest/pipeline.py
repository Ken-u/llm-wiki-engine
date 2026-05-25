"""Two-step CoT ingest pipeline.

Core flow (ported from autoIngest in ingest.ts):
  1. Cache check (SHA256)
  2. Document parse
  3. Step 1: LLM Analysis
  4. Step 2: LLM Generation
  5. Parse FILE blocks + safety check
  6. Write files (with merge)
  7. Trigger embedding
  8. Update cache
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiofiles

from app.documents.parser import parse_document
from app.ingest.cache import check_cache, update_cache
from app.ingest.parser import parse_file_blocks
from app.ingest.prompts import build_analysis_prompt, build_generation_prompt
from app.ingest.writer import write_file_blocks
from app.llm import client as llm_client

logger = logging.getLogger(__name__)


async def _read_file_safe(path: Path) -> str:
    if not path.exists():
        return ""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


async def auto_ingest(
    project_dir: str,
    source_path: str,
    *,
    on_progress: callable | None = None,
) -> list[str]:
    """Run the full two-step ingest pipeline for a single source document.

    Returns list of wiki paths written.
    """
    base = Path(project_dir)
    src = Path(source_path)

    source_identity = src.name

    # Parse document content
    if on_progress:
        on_progress("Parsing document...")
    content = parse_document(src)
    if not content.strip():
        logger.warning("Empty content from %s, skipping", source_path)
        return []

    # Cache check
    if await check_cache(project_dir, source_identity, content):
        logger.info("Cache hit for %s, skipping ingest", source_identity)
        return []

    # Read project context files
    purpose = await _read_file_safe(base / "purpose.md")
    index = await _read_file_safe(base / "wiki" / "index.md")
    overview = await _read_file_safe(base / "wiki" / "overview.md")
    schema = await _read_file_safe(base / "schema.md")

    # Truncate content if too large (leave room for prompts)
    max_content_chars = 100_000
    truncated = content[:max_content_chars] if len(content) > max_content_chars else content

    # Step 1: Analysis
    if on_progress:
        on_progress("Step 1/2: Analyzing source document...")
    analysis = await llm_client.stream_collect(
        build_analysis_prompt(purpose, index, truncated),
        f"Analyze this source document:\n\n**File:** {source_identity}\n\n---\n\n{truncated}",
        temperature=0.1,
        max_tokens=4096,
    )

    if not analysis.strip():
        raise RuntimeError(f"Analysis returned empty for {source_identity}")

    # Step 2: Generation
    if on_progress:
        on_progress("Step 2/2: Generating wiki pages...")

    source_base = source_identity.rsplit(".", 1)[0] if "." in source_identity else source_identity
    summary_path = f"wiki/sources/{source_base}.md"

    user_msg = "\n".join([
        f"Source document to process: **{source_identity}**",
        "",
        "The Stage 1 analysis below is CONTEXT to inform your output. Do NOT echo",
        "its tables, bullet points, or prose. Your output must be FILE/REVIEW",
        "blocks as specified in the system prompt — nothing else.",
        "",
        "## Stage 1 Analysis (context only — do not repeat)",
        "",
        analysis,
        "",
        "## Original Source Content",
        "",
        truncated,
        "",
        "---",
        "",
        f"Now emit the FILE blocks for the wiki files derived from **{source_identity}**.",
        "Your response MUST begin with `---FILE:` as the very first characters.",
        "No preamble. No analysis prose. Start immediately.",
    ])

    generation = await llm_client.stream_collect(
        build_generation_prompt(schema, purpose, index, source_identity, overview, truncated, summary_path),
        user_msg,
        temperature=0.1,
        max_tokens=8192,
    )

    if not generation.strip():
        raise RuntimeError(f"Generation returned empty for {source_identity}")

    # Parse FILE blocks
    result = parse_file_blocks(generation)
    for w in result.warnings:
        logger.warning("Parse warning: %s", w)

    if not result.blocks:
        logger.warning("No FILE blocks parsed from generation for %s", source_identity)
        return []

    # Write files
    if on_progress:
        on_progress(f"Writing {len(result.blocks)} files...")
    written = await write_file_blocks(project_dir, result.blocks, source_identity)

    # Trigger embedding for written pages (async, don't block)
    try:
        from app.embedding.service import embed_pages
        await embed_pages(project_dir, written)
    except Exception:
        logger.exception("Embedding failed for %s (non-fatal)", source_identity)

    # Update cache
    await update_cache(project_dir, source_identity, content, written)

    logger.info("Ingest complete for %s: %d files written", source_identity, len(written))
    return written
