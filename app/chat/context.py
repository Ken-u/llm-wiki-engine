"""Context budget allocation for Chat RAG.

Ported from context-budget.ts — dynamically allocates the LLM context
window across: system prompt, index, wiki page content, history, and
response reserve.
"""

from __future__ import annotations

from dataclasses import dataclass

CHARS_PER_TOKEN = 4  # rough estimate


@dataclass
class ContextBudget:
    total_tokens: int
    system_tokens: int
    index_tokens: int
    page_tokens: int
    history_tokens: int
    response_reserve: int


def compute_context_budget(max_context_size: int) -> ContextBudget:
    """Compute token budget allocation for chat RAG.

    Allocation strategy:
      - 15% response reserve
      - 5% index
      - 10% system prompt + instructions
      - 20% history
      - 50% page content
    """
    total = max_context_size
    response_reserve = int(total * 0.15)
    available = total - response_reserve
    system_tokens = int(available * 0.12)
    index_tokens = int(available * 0.06)
    history_tokens = int(available * 0.25)
    page_tokens = available - system_tokens - index_tokens - history_tokens

    return ContextBudget(
        total_tokens=total,
        system_tokens=system_tokens,
        index_tokens=index_tokens,
        page_tokens=page_tokens,
        history_tokens=history_tokens,
        response_reserve=response_reserve,
    )


def truncate_to_budget(text: str, token_budget: int) -> str:
    """Truncate text to fit within a token budget (approximate)."""
    max_chars = token_budget * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated to fit context budget ...]"
