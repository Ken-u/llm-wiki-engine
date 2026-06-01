"""Unit tests for the feedback evaluator: schema, parsing, consistency checks,
evidence override, and prompt building."""

import json
import pytest

from app.feedback.evaluator import (
    SUBMIT_EVALUATION_TOOL,
    _build_evaluator_prompt,
    _parse_tool_call,
    check_consistency,
    apply_evidence_override,
    EvaluatorInput,
    EvaluatorOutput,
)
from app.feedback.llm import ToolCallResult, SchemaValidationError, _validate_tools_schema
from app.feedback.queue import _extract_reads, _has_raw_usage


# ===================================================================
# 1. Tool schema validation
# ===================================================================

class TestSubmitEvaluationTool:
    def test_tool_type(self):
        assert SUBMIT_EVALUATION_TOOL["type"] == "function"

    def test_function_name(self):
        assert SUBMIT_EVALUATION_TOOL["function"]["name"] == "submit_evaluation"

    def test_has_all_evidence_fields(self):
        props = SUBMIT_EVALUATION_TOOL["function"]["parameters"]["properties"]
        for field in (
            "wiki_search_sufficient",
            "used_raw_read",
            "raw_read_required_for_answer",
            "answer_contains_info_missing_from_wiki",
            "answer_contradicts_wiki",
        ):
            assert field in props, f"Missing evidence field: {field}"

    def test_has_verdict_fields(self):
        props = SUBMIT_EVALUATION_TOOL["function"]["parameters"]["properties"]
        assert "needs_repair" in props
        assert "confidence" in props
        assert "reason" in props

    def test_required_params_complete(self):
        params = SUBMIT_EVALUATION_TOOL["function"]["parameters"]
        assert set(params["required"]) == {
            "wiki_search_sufficient",
            "used_raw_read",
            "raw_read_required_for_answer",
            "answer_contains_info_missing_from_wiki",
            "answer_contradicts_wiki",
            "needs_repair",
            "confidence",
            "reason",
        }

    def test_confidence_enum(self):
        props = SUBMIT_EVALUATION_TOOL["function"]["parameters"]["properties"]
        assert props["confidence"]["enum"] == ["high", "medium", "low"]


class TestSchemaValidation:
    def test_valid_schema_passes(self):
        _validate_tools_schema([SUBMIT_EVALUATION_TOOL])

    def test_empty_parameters_raises(self):
        bad_tool = {
            "type": "function",
            "function": {
                "name": "submit_evaluation",
                "parameters": {},
            },
        }
        with pytest.raises(SchemaValidationError, match="empty parameters"):
            _validate_tools_schema([bad_tool])

    def test_missing_properties_raises(self):
        bad_tool = {
            "type": "function",
            "function": {
                "name": "submit_evaluation",
                "parameters": {"type": "object"},
            },
        }
        with pytest.raises(SchemaValidationError, match="empty parameters"):
            _validate_tools_schema([bad_tool])


# ===================================================================
# 2. Prompt building
# ===================================================================

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

    def test_system_prompt_no_hard_raw_read_rule(self):
        inp = _make_input()
        msgs = _build_evaluator_prompt(inp)
        sys_prompt = msgs[0]["content"]
        assert "不要因为" in sys_prompt
        assert "风险信号" in sys_prompt
        assert "直接判定" in sys_prompt


# ===================================================================
# 3. Parsing & normalization
# ===================================================================

class TestParseToolCall:
    def test_full_evidence_output(self):
        tc = _tc(
            wiki_search_sufficient=True,
            used_raw_read=True,
            raw_read_required_for_answer=False,
            answer_contains_info_missing_from_wiki=False,
            answer_contradicts_wiki=False,
            needs_repair=False,
            confidence="high",
            reason="wiki 检索充足，raw_read 仅做验证",
        )
        result = _parse_tool_call(tc)
        assert result.needs_repair is False
        assert result.confidence == "high"
        assert result.raw["wiki_search_sufficient"] is True
        assert result.raw["raw_read_required_for_answer"] is False

    def test_strips_parameter_artifacts(self):
        tc = _tc(
            needs_repair=False,
            confidence="high\n</parameter",
            reason="wiki 内容充分。\n</parameter",
        )
        result = _parse_tool_call(tc)
        assert result.confidence == "high"
        assert result.reason == "wiki 内容充分。"

    def test_invalid_confidence_defaults_to_low(self):
        tc = _tc(needs_repair=True, confidence="certain", reason="bad")
        result = _parse_tool_call(tc)
        assert result.confidence == "low"


# ===================================================================
# 4. Consistency checks
# ===================================================================

class TestCheckConsistency:
    def test_consistent_no_repair(self):
        """wiki hit sufficient + no raw_read => no repair: consistent"""
        out = _output(
            needs_repair=False, confidence="high",
            reason="wiki 检索充足，无需修复",
            wiki_search_sufficient=True,
            raw_read_required_for_answer=False,
            answer_contains_info_missing_from_wiki=False,
            answer_contradicts_wiki=False,
        )
        assert check_consistency(out) is None

    def test_consistent_needs_repair(self):
        """wiki miss + raw_read required => needs repair: consistent"""
        out = _output(
            needs_repair=True, confidence="high",
            reason="Agent 依赖 read_raw 才能回答，wiki 检索不足",
            wiki_search_sufficient=False,
            raw_read_required_for_answer=True,
            answer_contains_info_missing_from_wiki=True,
            answer_contradicts_wiki=False,
        )
        assert check_consistency(out) is None

    def test_no_repair_but_reason_says_repair(self):
        """needs_repair=false but reason mentions repair keywords"""
        out = _output(
            needs_repair=False, confidence="medium",
            reason="wiki 内容缺失，需要修复",
        )
        conflict = check_consistency(out)
        assert conflict is not None
        assert "缺失" in conflict or "needs_repair=false" in conflict

    def test_needs_repair_but_reason_says_no_issue(self):
        """needs_repair=true but reason says all is fine"""
        out = _output(
            needs_repair=True, confidence="medium",
            reason="wiki 内容充足，无明显缺陷",
        )
        conflict = check_consistency(out)
        assert conflict is not None

    def test_no_repair_but_multiple_evidence_signals(self):
        """needs_repair=false but 2+ evidence signals say otherwise"""
        out = _output(
            needs_repair=False, confidence="medium",
            reason="一切正常",
            raw_read_required_for_answer=True,
            answer_contains_info_missing_from_wiki=True,
            answer_contradicts_wiki=False,
        )
        conflict = check_consistency(out)
        assert conflict is not None
        assert "多项证据" in conflict

    def test_high_confidence_needs_repair_but_wiki_sufficient(self):
        """needs_repair=true + high confidence but wiki is sufficient"""
        out = _output(
            needs_repair=True, confidence="high",
            reason="需要补充内容",
            wiki_search_sufficient=True,
            raw_read_required_for_answer=False,
        )
        conflict = check_consistency(out)
        assert conflict is not None
        assert "high confidence" in conflict or "high" in conflict

    def test_wiki_hit_plus_raw_read_verify_is_no_repair(self):
        """wiki sufficient + raw_read only for verification => no repair, consistent"""
        out = _output(
            needs_repair=False, confidence="high",
            reason="wiki 检索充足，raw_read 仅做验证",
            wiki_search_sufficient=True,
            used_raw_read=True,
            raw_read_required_for_answer=False,
            answer_contains_info_missing_from_wiki=False,
        )
        assert check_consistency(out) is None


# ===================================================================
# 5. Evidence override
# ===================================================================

class TestApplyEvidenceOverride:
    def test_override_repair_to_no_repair(self):
        """Model says repair but all evidence says no"""
        out = _output(
            needs_repair=True, confidence="high",
            reason="需要修复",
            wiki_search_sufficient=True,
            raw_read_required_for_answer=False,
            answer_contains_info_missing_from_wiki=False,
            answer_contradicts_wiki=False,
        )
        result = apply_evidence_override(out)
        assert result.needs_repair is False
        assert result.raw.get("_overridden") is True

    def test_override_no_repair_to_repair_strong_signals(self):
        """Model says no repair but raw_read required + missing info"""
        out = _output(
            needs_repair=False, confidence="medium",
            reason="正常",
            raw_read_required_for_answer=True,
            answer_contains_info_missing_from_wiki=True,
        )
        result = apply_evidence_override(out)
        assert result.needs_repair is True
        assert result.raw.get("_overridden") is True

    def test_override_no_repair_when_contradicts_wiki(self):
        """Model says no repair but answer contradicts wiki"""
        out = _output(
            needs_repair=False, confidence="low",
            reason="正常",
            answer_contradicts_wiki=True,
        )
        result = apply_evidence_override(out)
        assert result.needs_repair is True

    def test_no_override_when_consistent(self):
        """Consistent output is not modified"""
        out = _output(
            needs_repair=False, confidence="high",
            reason="wiki 检索充足",
            wiki_search_sufficient=True,
            raw_read_required_for_answer=False,
            answer_contains_info_missing_from_wiki=False,
            answer_contradicts_wiki=False,
        )
        result = apply_evidence_override(out)
        assert result.needs_repair is False
        assert result.raw.get("_overridden") is None


# ===================================================================
# 6. Existing queue helper tests (unchanged)
# ===================================================================

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


# ===================================================================
# Helpers
# ===================================================================

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


def _tc(**kwargs) -> ToolCallResult:
    defaults = {
        "wiki_search_sufficient": True,
        "used_raw_read": False,
        "raw_read_required_for_answer": False,
        "answer_contains_info_missing_from_wiki": False,
        "answer_contradicts_wiki": False,
        "needs_repair": False,
        "confidence": "low",
        "reason": "",
    }
    defaults.update(kwargs)
    return ToolCallResult(id="tc_test", name="submit_evaluation", arguments=defaults)


def _output(*, needs_repair: bool, confidence: str, reason: str, **evidence) -> EvaluatorOutput:
    raw = {
        "wiki_search_sufficient": evidence.get("wiki_search_sufficient", True),
        "used_raw_read": evidence.get("used_raw_read", False),
        "raw_read_required_for_answer": evidence.get("raw_read_required_for_answer", False),
        "answer_contains_info_missing_from_wiki": evidence.get("answer_contains_info_missing_from_wiki", False),
        "answer_contradicts_wiki": evidence.get("answer_contradicts_wiki", False),
        "needs_repair": needs_repair,
        "confidence": confidence,
        "reason": reason,
    }
    return EvaluatorOutput(
        needs_repair=needs_repair,
        confidence=confidence,
        reason=reason,
        raw=raw,
    )
