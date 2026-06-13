"""Tests for agent context compression."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.context_compress import (
    HISTORY_SUMMARY_HEADER,
    _parse_layout,
    compress_agent_context,
    estimate_prompt_tokens,
    resolve_prompt_tokens,
    should_compress,
)
from app.agents.tools import ToolContext
from app.llm.client import LLMResponse, TokenUsage, ToolCallRequest


def test_should_compress_at_threshold():
    max_ctx = 128000
    threshold = 0.85
    assert not should_compress(int(max_ctx * 0.84), max_ctx, threshold)
    assert should_compress(int(max_ctx * 0.85), max_ctx, threshold)
    assert should_compress(int(max_ctx * 0.90), max_ctx, threshold)


def test_resolve_prompt_tokens_prefers_api():
    tokens, source = resolve_prompt_tokens([], api_prompt_tokens=90000)
    assert tokens == 90000
    assert source == "api"


def test_resolve_prompt_tokens_estimates_when_missing():
    messages = [{"role": "user", "content": "a" * 400}]
    tokens, source = resolve_prompt_tokens(messages, api_prompt_tokens=None)
    assert tokens == 100
    assert source == "estimated"


def test_parse_layout_with_history_and_tools():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "h1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "h2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "h3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "h4"},
        {"role": "assistant", "content": "a4"},
        {"role": "user", "content": "h5"},
        {"role": "assistant", "content": "a5"},
        {"role": "user", "content": "current"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "search_wiki", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "1", "content": '{"results": []}'},
    ]
    layout = _parse_layout(messages)
    assert layout is not None
    assert layout.history_end == 11  # index of "current" user
    assert layout.current_user_idx == 11
    assert layout.tool_tail_start == 12


@pytest.mark.asyncio
async def test_compress_history_keeps_recent_two_turns():
    history = [
        {"role": "user", "content": f"old user {i}"}
        if i % 2 == 0
        else {"role": "assistant", "content": f"old assistant {i}"}
        for i in range(12)
    ]
    messages = [{"role": "system", "content": "sys"}] + history + [
        {"role": "user", "content": "current question"},
    ]
    persisted = list(history)

    with patch(
        "app.agents.context_compress.llm_client.complete",
        new=AsyncMock(return_value="摘要：讨论了旧话题。"),
    ):
        result = await compress_agent_context(
            messages,
            persisted,
            prompt_tokens=110000,
            max_context_size=128000,
            threshold=0.85,
            target=0.65,
        )

    assert result.compressed
    assert HISTORY_SUMMARY_HEADER in result.messages[0]["content"]
    assert "摘要：讨论了旧话题。" in result.messages[0]["content"]
    # 12 history -> keep last 4, compress first 8
    history_in_messages = [
        m for m in result.messages[1:-1]
        if m.get("role") in ("user", "assistant")
    ]
    assert len(history_in_messages) == 4
    assert result.messages[-1]["content"] == "current question"
    assert len(result.persisted_history) == 4


@pytest.mark.asyncio
async def test_compress_no_op_below_threshold():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    result = await compress_agent_context(
        messages,
        [],
        prompt_tokens=1000,
        max_context_size=128000,
        threshold=0.85,
        target=0.65,
    )
    assert not result.compressed
    assert result.messages == messages


@pytest.mark.asyncio
async def test_orchestrator_emits_compressed_history_on_high_usage():
    from app.agents.orchestrator import run_agent_turn

    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "q4"},
        {"role": "assistant", "content": "a4"},
        {"role": "user", "content": "q5"},
        {"role": "assistant", "content": "a5"},
    ]

    tool_resp = LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(id="tc1", name="search_wiki", arguments='{"query":"test"}'),
        ],
        finish_reason="tool_calls",
        usage=TokenUsage(prompt_tokens=110000, completion_tokens=10, total_tokens=110010),
    )
    final_resp = LLMResponse(
        content="final answer",
        tool_calls=[],
        finish_reason="stop",
        usage=TokenUsage(prompt_tokens=5000, completion_tokens=20, total_tokens=5020),
    )

    ctx = ToolContext(main_projects=[], ticket_project=None)

    with patch(
        "app.agents.orchestrator.complete_with_tools",
        new=AsyncMock(side_effect=[tool_resp, final_resp]),
    ), patch(
        "app.agents.orchestrator.execute_tool",
        new=AsyncMock(return_value={"results": []}),
    ), patch(
        "app.agents.context_compress.llm_client.complete",
        new=AsyncMock(return_value="历史摘要"),
    ):
        events = []
        async for event in run_agent_turn("current", history, "", ctx, max_tool_calls=5):
            events.append(json.loads(event))

    done = next(e for e in events if e.get("done"))
    assert done["context_compressed"] is True
    assert "compressed_history" in done
    assert len(done["compressed_history"]) == 4
    compress_calls = [e for e in events if e.get("tool_call", {}).get("name") == "compress_context"]
    compress_results = [e for e in events if e.get("tool_result", {}).get("name") == "compress_context"]
    assert len(compress_calls) >= 1
    assert len(compress_results) >= 1
    assert compress_results[0]["tool_result"]["result"]["compressed"] is True
    assert any(e.get("token") == "final answer" for e in events)


def test_conversations_append_turn_with_compressed_history(tmp_path):
    from app.agents.conversations import append_turn, get_conversation

    append_turn(
        tmp_path,
        agent_id="agent-1",
        user_id=1,
        conversation_id="conv-1",
        user_message="first",
        assistant_answer="answer 1",
    )
    compressed = [
        {"role": "user", "content": "kept q"},
        {"role": "assistant", "content": "kept a"},
    ]
    append_turn(
        tmp_path,
        agent_id="agent-1",
        user_id=1,
        conversation_id="conv-1",
        user_message="second",
        assistant_answer="answer 2",
        compressed_history=compressed,
    )
    loaded = get_conversation(
        tmp_path, agent_id="agent-1", user_id=1, conversation_id="conv-1",
    )
    assert loaded is not None
    assert len(loaded["messages"]) == 4
    assert loaded["messages"][0]["content"] == "kept q"
    assert loaded["messages"][-1]["content"] == "answer 2"
