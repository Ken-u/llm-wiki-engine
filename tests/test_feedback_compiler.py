"""Unit tests for compiler tool definition and prompt building."""

from app.feedback.compiler import (
    SUBMIT_REPAIR_TOOL,
    _build_compiler_prompt,
    _parse_tool_call,
    CompilerInput,
)
from app.feedback.llm import ToolCallResult


class TestSubmitRepairTool:
    def test_tool_type(self):
        assert SUBMIT_REPAIR_TOOL["type"] == "function"

    def test_function_name(self):
        assert SUBMIT_REPAIR_TOOL["function"]["name"] == "submit_repair"

    def test_required_params(self):
        params = SUBMIT_REPAIR_TOOL["function"]["parameters"]
        assert set(params["required"]) == {"proposed_content", "change_summary", "confidence"}

    def test_confidence_enum(self):
        props = SUBMIT_REPAIR_TOOL["function"]["parameters"]["properties"]
        assert props["confidence"]["enum"] == ["high", "medium", "low"]

    def test_has_proposed_content(self):
        props = SUBMIT_REPAIR_TOOL["function"]["parameters"]["properties"]
        assert "proposed_content" in props
        assert props["proposed_content"]["type"] == "string"


class TestBuildCompilerPrompt:
    def test_returns_two_messages(self):
        inp = _make_input()
        msgs = _build_compiler_prompt(inp)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_includes_target_path(self):
        inp = _make_input(target_page_path="wiki/test.md")
        msgs = _build_compiler_prompt(inp)
        assert "wiki/test.md" in msgs[1]["content"]

    def test_includes_existing_content(self):
        inp = _make_input(existing_page_content="# Title\nSome content")
        msgs = _build_compiler_prompt(inp)
        assert "# Title" in msgs[1]["content"]

    def test_new_page_note(self):
        inp = _make_input(existing_page_content=None)
        msgs = _build_compiler_prompt(inp)
        assert "需新建" in msgs[1]["content"]

    def test_review_guidance_included(self):
        inp = _make_input(review_guidance="请修正第三段")
        msgs = _build_compiler_prompt(inp)
        assert "请修正第三段" in msgs[1]["content"]


class TestParseToolCall:
    def test_full_args(self):
        tc = ToolCallResult(
            name="submit_repair",
            arguments={
                "proposed_content": "# Fixed\nNew content",
                "change_summary": "Fixed typo",
                "confidence": "high",
                "sections_modified": ["section1"],
            },
        )
        result = _parse_tool_call(tc)
        assert result.proposed_content == "# Fixed\nNew content"
        assert result.change_summary == "Fixed typo"
        assert result.confidence == "high"
        assert result.sections_modified == ["section1"]

    def test_minimal_args(self):
        tc = ToolCallResult(
            name="submit_repair",
            arguments={
                "proposed_content": "content",
                "change_summary": "fix",
                "confidence": "low",
            },
        )
        result = _parse_tool_call(tc)
        assert result.sections_modified == []


def _make_input(**kwargs) -> CompilerInput:
    defaults = {
        "user_message": "test",
        "assistant_answer": "answer",
        "evaluator_result": {"needs_repair": True},
        "target_page_path": "wiki/test.md",
        "existing_page_content": "# Test\nOld content",
        "review_guidance": None,
        "wiki_reads": [],
        "raw_reads": [],
    }
    defaults.update(kwargs)
    return CompilerInput(**defaults)
