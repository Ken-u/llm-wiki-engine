"""Two-step CoT ingest pipeline with step-level checkpointing.

Core flow:
  1. Cache check (SHA256)
  2. Document parse
  3. Step 1: LLM Analysis  → checkpoint analysis to disk
  4. Step 2: LLM Generation → checkpoint generation to disk
  5. Parse FILE blocks + safety check
  6. Write files (with merge)
  7. Trigger embedding
  8. Update cache + clean checkpoints

Checkpoints allow resuming from the last completed step after a crash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import aiofiles

from app.documents.parser import parse_document
from app.ingest.cache import check_cache, update_cache
from app.ingest.knowledge_tool import collect_with_main_knowledge_tools, get_case_library_main_projects
from app.ingest.parser import parse_file_blocks
from app.ingest.prompts import build_analysis_prompt, build_generation_prompt
from app.ingest.writer import write_file_blocks

logger = logging.getLogger(__name__)


async def _read_file_safe(path: Path) -> str:
    if not path.exists():
        return ""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


def _checkpoint_dir(project_dir: str) -> Path:
    d = Path(project_dir) / ".llm-wiki" / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_key(source_path: str) -> str:
    return hashlib.sha256(source_path.encode()).hexdigest()[:16]


async def _save_checkpoint(project_dir: str, source_path: str, step: int, data: dict) -> None:
    cp_dir = _checkpoint_dir(project_dir)
    key = _checkpoint_key(source_path)
    payload = {"step": step, "source": source_path, **data}
    async with aiofiles.open(cp_dir / f"{key}.json", "w", encoding="utf-8") as f:
        await f.write(json.dumps(payload, ensure_ascii=False))


async def _load_checkpoint(project_dir: str, source_path: str) -> dict | None:
    cp_dir = _checkpoint_dir(project_dir)
    key = _checkpoint_key(source_path)
    cp_file = cp_dir / f"{key}.json"
    if not cp_file.exists():
        return None
    try:
        async with aiofiles.open(cp_file, "r", encoding="utf-8") as f:
            return json.loads(await f.read())
    except Exception:
        return None


async def _remove_checkpoint(project_dir: str, source_path: str) -> None:
    cp_dir = _checkpoint_dir(project_dir)
    key = _checkpoint_key(source_path)
    cp_file = cp_dir / f"{key}.json"
    cp_file.unlink(missing_ok=True)


async def _snapshot_targets(project_dir: str, paths: list[str]) -> dict[str, str | None]:
    base = Path(project_dir)
    snapshot: dict[str, str | None] = {}
    for path in paths:
        target = base / path
        if target.exists():
            async with aiofiles.open(target, "r", encoding="utf-8") as f:
                snapshot[path] = await f.read()
        else:
            snapshot[path] = None
    return snapshot


async def _restore_targets(project_dir: str, snapshot: dict[str, str | None]) -> None:
    base = Path(project_dir)
    for path, content in snapshot.items():
        target = base / path
        if content is None:
            target.unlink(missing_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(target, "w", encoding="utf-8") as f:
            await f.write(content)


class ProgressCallback:
    """Wraps the on_progress callable and step tracking."""

    def __init__(self, on_progress=None, on_step=None):
        self.on_progress = on_progress
        self.on_step = on_step

    async def report(self, msg: str):
        if self.on_progress:
            await self.on_progress(msg)

    async def set_step(self, step: int):
        if self.on_step:
            await self.on_step(step)


async def _ensure_not_paused(should_pause) -> None:
    if should_pause:
        await should_pause()


async def auto_ingest(
    project_dir: str,
    source_path: str,
    *,
    project_id: str | None = None,
    on_progress=None,
    on_step=None,
    resume_step: int = 0,
    should_pause=None,
) -> list[str]:
    """Run the full two-step ingest pipeline for a single source document.

    Args:
        resume_step: step already completed (from DB). 0=fresh start.
        on_step: callback(step_int) to persist step to DB.

    Returns list of wiki paths written.
    """
    cb = ProgressCallback(on_progress, on_step)
    base = Path(project_dir)
    src = Path(source_path)
    source_identity = src.name

    # ── Parse document ──
    await _ensure_not_paused(should_pause)
    await cb.report("Parsing document...")
    content = parse_document(src)
    if not content.strip():
        logger.warning("Empty content from %s, skipping", source_path)
        return []

    # ── Cache check ──
    if await check_cache(project_dir, source_identity, content):
        logger.info("Cache hit for %s, skipping ingest", source_identity)
        await _remove_checkpoint(project_dir, source_path)
        return []

    # Load existing checkpoint
    checkpoint = await _load_checkpoint(project_dir, source_path)
    cp_step = checkpoint.get("step", 0) if checkpoint else 0
    effective_step = max(resume_step, cp_step)

    # Read project context
    purpose = await _read_file_safe(base / "purpose.md")
    index = await _read_file_safe(base / "wiki" / "index.md")
    overview = await _read_file_safe(base / "wiki" / "overview.md")
    schema = await _read_file_safe(base / "schema.md")
    main_knowledge_projects = await get_case_library_main_projects(project_id) if project_id else []
    if main_knowledge_projects:
        await cb.report(f"Linked main knowledge lookup enabled ({len(main_knowledge_projects)} project(s))")

    max_content_chars = 100_000
    truncated = content[:max_content_chars] if len(content) > max_content_chars else content

    # ── Step 1: Analysis ──
    if effective_step < 1:
        await _ensure_not_paused(should_pause)
        await cb.report("Step 1/2: Analyzing source document...")
        analysis = await collect_with_main_knowledge_tools(
            build_analysis_prompt(purpose, index, truncated),
            f"Analyze this source document:\n\n**File:** {source_identity}\n\n---\n\n{truncated}",
            main_knowledge_projects,
            temperature=0.1,
            max_tokens=4096,
        )
        if not analysis.strip():
            raise RuntimeError(f"Analysis returned empty for {source_identity}")

        await _save_checkpoint(project_dir, source_path, 1, {"analysis": analysis})
        await cb.set_step(1)
        logger.info("Step 1 complete for %s, checkpoint saved", source_identity)
    else:
        analysis = checkpoint.get("analysis", "") if checkpoint else ""
        if not analysis:
            raise RuntimeError(f"Checkpoint missing analysis for {source_identity}, re-run from scratch")
        await cb.report("Step 1/2: Resuming from checkpoint (analysis cached)")
        logger.info("Resuming %s from step %d (analysis from checkpoint)", source_identity, effective_step)

    # ── Step 2: Generation ──
    if effective_step < 2:
        await _ensure_not_paused(should_pause)
        await cb.report("Step 2/2: Generating wiki pages...")

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

        generation = await collect_with_main_knowledge_tools(
            build_generation_prompt(schema, purpose, index, source_identity, overview, truncated, summary_path),
            user_msg,
            main_knowledge_projects,
            temperature=0.1,
            max_tokens=8192,
        )
        if not generation.strip():
            raise RuntimeError(f"Generation returned empty for {source_identity}")

        await _save_checkpoint(project_dir, source_path, 2, {"analysis": analysis, "generation": generation})
        await cb.set_step(2)
        logger.info("Step 2 complete for %s, checkpoint saved", source_identity)
    else:
        generation = checkpoint.get("generation", "") if checkpoint else ""
        if not generation:
            raise RuntimeError(f"Checkpoint missing generation for {source_identity}, re-run from scratch")
        await cb.report("Step 2/2: Resuming from checkpoint (generation cached)")
        logger.info("Resuming %s from step %d (generation from checkpoint)", source_identity, effective_step)

    # ── Parse + Write ──
    await _ensure_not_paused(should_pause)
    result = parse_file_blocks(generation)
    for w in result.warnings:
        logger.warning("Parse warning: %s", w)

    if not result.blocks:
        logger.warning("No FILE blocks parsed from generation for %s", source_identity)
        await _remove_checkpoint(project_dir, source_path)
        return []

    await cb.report(f"Writing {len(result.blocks)} files...")
    target_snapshot = await _snapshot_targets(project_dir, [block.path for block in result.blocks])
    try:
        written = await write_file_blocks(project_dir, result.blocks, source_identity)
        await cb.set_step(3)

        # ── Embedding ──
        try:
            await _ensure_not_paused(should_pause)
            from app.embedding.service import embed_pages
            await cb.report("Embedding wiki pages...")
            await embed_pages(project_dir, written)
        except Exception:
            logger.exception("Embedding failed for %s (non-fatal)", source_identity)

        # ── Finalize ──
        await update_cache(project_dir, source_identity, content, written)
        await _remove_checkpoint(project_dir, source_path)
    except Exception:
        await _restore_targets(project_dir, target_snapshot)
        raise

    logger.info("Ingest complete for %s: %d files written", source_identity, len(written))
    return written
