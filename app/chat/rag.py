"""4-phase RAG pipeline for Chat, ported from chat-panel.tsx.

Phase 1: Hybrid search (top-10)
Phase 2: Knowledge graph 1-hop expansion
Phase 3: Context budget allocation + page loading
Phase 4: System prompt assembly
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import AsyncGenerator

import aiofiles

from app.chat.context import compute_context_budget, truncate_to_budget
from app.config import get_config
from app.llm import client as llm_client
from app.search.bm25 import search_bm25
from app.search.fusion import rrf_fusion
from app.search.vector import search_vector
from app.wiki.graph import build_wiki_graph, graph_expand

logger = logging.getLogger(__name__)


async def _read_file(path: Path) -> str:
    if not path.exists():
        return ""
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        return await f.read()


async def chat_rag(
    project_dir: str,
    message: str,
    history: list[dict],
    *,
    conversation_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Full RAG chat pipeline with streaming response."""
    cfg = get_config()
    base = Path(project_dir)

    # Phase 1: Hybrid search
    kw_results = search_bm25(project_dir, message, top_k=20)
    vec_results = await search_vector(project_dir, message, top_k=20)
    fused = rrf_fusion(kw_results, vec_results, message)[:10]

    seed_ids = [r.page_id for r in fused]

    # Phase 2: Graph expansion (1-hop)
    graph = build_wiki_graph(project_dir)
    expanded_ids = graph_expand(seed_ids, graph, max_related=3)
    all_page_ids = list(dict.fromkeys(seed_ids + expanded_ids))

    # Phase 3: Context budget
    budget = compute_context_budget(cfg.llm.max_context_size)

    purpose = await _read_file(base / "purpose.md")
    index = await _read_file(base / "wiki" / "index.md")
    index_truncated = truncate_to_budget(index, budget.index_tokens)

    # Load page content within budget
    pages_content: list[str] = []
    total_chars = 0
    max_page_chars = budget.page_tokens * 4  # chars per token estimate

    for pid in all_page_ids:
        # Try to find the page file
        found = False
        for subdir in ["entities", "concepts", "sources", "queries", "comparisons", "synthesis"]:
            page_path = base / "wiki" / subdir / f"{pid}.md"
            if page_path.exists():
                content = await _read_file(page_path)
                if content and total_chars + len(content) <= max_page_chars:
                    pages_content.append(f"### {pid}\n\n{content}")
                    total_chars += len(content)
                found = True
                break

        if not found:
            # Try direct wiki/{pid}.md
            page_path = base / "wiki" / f"{pid}.md"
            if page_path.exists():
                content = await _read_file(page_path)
                if content and total_chars + len(content) <= max_page_chars:
                    pages_content.append(f"### {pid}\n\n{content}")
                    total_chars += len(content)

    # Phase 4: Assemble system prompt
    references = "\n\n---\n\n".join(pages_content) if pages_content else "No relevant wiki pages found."

    system_prompt = "\n".join([
        "You are a knowledgeable assistant for a research wiki. Answer the user's question",
        "based on the wiki content provided below. If the wiki doesn't contain relevant",
        "information, say so honestly.",
        "",
        "Guidelines:",
        "- Cite specific wiki pages when referencing information using [[page-name]] syntax",
        "- Be precise and factual — prefer wiki content over general knowledge",
        "- If multiple pages cover the topic, synthesize them",
        "- Acknowledge limitations or contradictions in the wiki if relevant",
        "",
        f"## Wiki Purpose\n{purpose}" if purpose else "",
        "",
        f"## Wiki Index\n{index_truncated}" if index_truncated else "",
        "",
        f"## Relevant Wiki Pages\n\n{references}",
    ])

    system_prompt = truncate_to_budget(system_prompt, budget.system_tokens + budget.page_tokens + budget.index_tokens)

    # Truncate history to fit budget
    history_truncated = history
    history_chars = sum(len(m.get("content", "")) for m in history)
    max_history_chars = budget.history_tokens * 4
    if history_chars > max_history_chars:
        # Keep only the most recent messages
        trimmed = []
        running = 0
        for m in reversed(history):
            c = len(m.get("content", ""))
            if running + c > max_history_chars:
                break
            trimmed.insert(0, m)
            running += c
        history_truncated = trimmed

    # Build messages
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_truncated)
    messages.append({"role": "user", "content": message})

    # Stream response
    async for token in llm_client.stream(
        messages,
        temperature=cfg.llm.chat_temperature,
        max_tokens=4096,
    ):
        yield token
