"""Tests for main knowledge lookup tools during case-library ingest."""

import asyncio
from unittest.mock import AsyncMock, patch

from app.ingest.knowledge_tool import (
    MAIN_KNOWLEDGE_LOOKUP_TOOL,
    collect_with_main_knowledge_tools,
)
from app.llm.client import LLMResponse, ToolCallRequest
from app.projects.models import Project


def test_main_knowledge_lookup_tool_requires_term():
    fn = MAIN_KNOWLEDGE_LOOKUP_TOOL["function"]
    assert fn["name"] == "lookup_main_knowledge_term"
    assert "term" in fn["parameters"]["required"]


def test_collect_without_main_projects_uses_plain_ingest_completion():
    async def run():
        with patch(
            "app.ingest.knowledge_tool.llm_client.stream_collect",
            AsyncMock(return_value="plain result"),
        ) as stream_collect:
            result = await collect_with_main_knowledge_tools(
                "system",
                "user",
                [],
                temperature=0.1,
                max_tokens=1024,
            )

        assert result == "plain result"
        stream_collect.assert_awaited_once_with(
            "system",
            "user",
            temperature=0.1,
            max_tokens=1024,
        )

    asyncio.run(run())


def test_collect_executes_main_knowledge_lookup_tool_call():
    main_project = Project(id="main-1", name="Main Wiki", slug="main", description="", created_by=1)
    first = LLMResponse(
        content=None,
        tool_calls=[
            ToolCallRequest(
                id="call-1",
                name="lookup_main_knowledge_term",
                arguments='{"term": "SLA", "question": "SLA 是什么？"}',
            )
        ],
        finish_reason="tool_calls",
    )
    second = LLMResponse(content="SLA means service level agreement.", tool_calls=[], finish_reason="stop")

    async def run():
        with patch(
            "app.ingest.knowledge_tool.llm_client.complete_with_tools",
            AsyncMock(side_effect=[first, second]),
        ) as complete_with_tools:
            with patch(
                "app.ingest.knowledge_tool.knowledge_lookup",
                AsyncMock(return_value="SLA 是服务等级协议。"),
            ) as lookup:
                result = await collect_with_main_knowledge_tools(
                    "system",
                    "user",
                    [main_project],
                    temperature=0.1,
                    max_tokens=1024,
                )

        assert result == "SLA means service level agreement."
        lookup.assert_awaited_once()
        assert lookup.await_args.args[0] is main_project
        assert "SLA" in lookup.await_args.args[1]
        assert complete_with_tools.await_count == 2
        second_messages = complete_with_tools.await_args_list[1].args[0]
        assert second_messages[-1]["role"] == "tool"
        assert "服务等级协议" in second_messages[-1]["content"]

    asyncio.run(run())
