"""Unit tests for evaluator tool definition and prompt building."""

import json

from app.feedback.evaluator import (
    SUBMIT_EVALUATION_TOOL,
    _build_evaluator_prompt,
    _parse_tool_call,
    EvaluatorInput,
)
from app.feedback.llm import ToolCallResult
from app.feedback.queue import _extract_reads, _has_raw_usage


class TestSubmitEvaluationTool:
    def test_tool_type(self):
        assert SUBMIT_EVALUATION_TOOL["type"] == "function"

    def test_function_name(self):
        assert SUBMIT_EVALUATION_TOOL["function"]["name"] == "submit_evaluation"

    def test_required_params(self):
        params = SUBMIT_EVALUATION_TOOL["function"]["parameters"]
        assert "needs_repair" in params["properties"]
        assert "confidence" in params["properties"]
        assert "reason" in params["properties"]
        assert set(params["required"]) == {"needs_repair", "confidence", "reason"}

    def test_confidence_enum(self):
        props = SUBMIT_EVALUATION_TOOL["function"]["parameters"]["properties"]
        assert props["confidence"]["enum"] == ["high", "medium", "low"]


class TestBuildEvaluatorPrompt:
    def test_returns_two_messages(self):
        inp = _make_input()
        msgs = _build_evaluator_prompt(inp)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_user_content_includes_question(self):
        inp = _make_input(user_message="测试问题")
        msgs = _build_evaluator_prompt(inp)
        assert "测试问题" in msgs[1]["content"]


class TestParseToolCall:
    def test_full_args(self):
        tc = ToolCallResult(
            id="tc_1",
            name="submit_evaluation",
            arguments={
                "needs_repair": True,
                "confidence": "high",
                "reason": "测试原因",
            },
        )
        result = _parse_tool_call(tc)
        assert result.needs_repair is True
        assert result.confidence == "high"
        assert result.reason == "测试原因"

    def test_minimal_args(self):
        tc = ToolCallResult(
            id="tc_2",
            name="submit_evaluation",
            arguments={"needs_repair": False, "confidence": "low", "reason": "ok"},
        )
        result = _parse_tool_call(tc)
        assert result.needs_repair is False
        assert result.confidence == "low"

    def test_strips_parameter_artifacts_from_args(self):
        tc = ToolCallResult(
            id="tc_3",
            name="submit_evaluation",
            arguments={
                "needs_repair": False,
                "confidence": "high\n</parameter",
                "reason": "wiki 内容充分。\n</parameter",
            },
        )
        result = _parse_tool_call(tc)
        assert result.needs_repair is False
        assert result.confidence == "high"
        assert result.reason == "wiki 内容充分。"
        assert result.raw == {
            "needs_repair": False,
            "confidence": "high",
            "reason": "wiki 内容充分。",
        }

    def test_invalid_confidence_defaults_to_low(self):
        tc = ToolCallResult(
            id="tc_4",
            name="submit_evaluation",
            arguments={
                "needs_repair": True,
                "confidence": "certain",
                "reason": "bad",
            },
        )
        result = _parse_tool_call(tc)
        assert result.confidence == "low"


class TestExtractReads:
    def test_splits_wiki_and_raw(self):
        traces = [
            {"name": "search_wiki", "arguments": {"query": "test"}},
            {"name": "read_raw", "arguments": {"path": "a.md"}},
            {"name": "read_wiki_page", "arguments": {"path": "x.md"}},
            {"name": "other_tool", "arguments": {}},
            {"name": "get_wiki_index", "arguments": {}},
        ]
        wiki, raw = _extract_reads(traces)
        assert len(wiki) == 3
        assert len(raw) == 1

    def test_empty(self):
        wiki, raw = _extract_reads([])
        assert wiki == []
        assert raw == []


class TestHasRawUsage:
    def test_true_with_read_raw(self):
        assert _has_raw_usage([{"name": "read_raw"}])

    def test_true_with_grep_raw(self):
        assert _has_raw_usage([{"name": "grep_raw"}])

    def test_false_with_wiki_only(self):
        assert not _has_raw_usage([{"name": "search_wiki"}])

    def test_false_empty(self):
        assert not _has_raw_usage([])


def _make_input(**kwargs) -> EvaluatorInput:
    defaults = {
        "user_message": "test question",
        "assistant_answer": "test answer",
        "tool_traces": [],
        "wiki_reads": [],
        "raw_reads": [],
    }
    defaults.update(kwargs)
    return EvaluatorInput(**defaults)
