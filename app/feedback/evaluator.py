"""Feedback evaluator agent — evidence-based evaluation with consistency checks.

The evaluator outputs *evidence fields* first, then a final verdict.
Post-processing validates consistency between evidence and verdict,
auto-retrying once on conflict with an explanation of the mismatch.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from app.config import FeedbackModelConfig
from app.feedback.llm import ToolCallResult, complete_with_tools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool schema — evidence + verdict
# ---------------------------------------------------------------------------

SUBMIT_EVALUATION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_evaluation",
        "description": (
            "提交 wiki 质量评估结果。先输出证据字段，再给出最终判定。"
            "reason 必须与 needs_repair 保持一致。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "wiki_search_sufficient": {
                    "type": "boolean",
                    "description": "wiki 检索结果是否足以支持 Agent 的回答核心内容。",
                },
                "used_raw_read": {
                    "type": "boolean",
                    "description": "Agent 是否调用了 read_raw 或 grep_raw。",
                },
                "raw_read_required_for_answer": {
                    "type": "boolean",
                    "description": "回答的核心内容是否依赖 raw_read 结果（而非仅做验证或补全）。",
                },
                "answer_contains_info_missing_from_wiki": {
                    "type": "boolean",
                    "description": "回答中是否包含 wiki 检索结果中不存在的关键信息。",
                },
                "answer_contradicts_wiki": {
                    "type": "boolean",
                    "description": "回答内容是否与 wiki 现有内容矛盾。",
                },
                "needs_repair": {
                    "type": "boolean",
                    "description": "最终判定：wiki 是否需要修复。",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "对当前判定的置信度（不是问题严重程度）。",
                },
                "reason": {
                    "type": "string",
                    "description": "一句话判定原因，必须与 needs_repair 一致。",
                },
            },
            "required": [
                "wiki_search_sufficient",
                "used_raw_read",
                "raw_read_required_for_answer",
                "answer_contains_info_missing_from_wiki",
                "answer_contradicts_wiki",
                "needs_repair",
                "confidence",
                "reason",
            ],
        },
    },
}

# ---------------------------------------------------------------------------
# System prompt — raw_read is a weak signal, not a hard rule
# ---------------------------------------------------------------------------

EVALUATOR_SYSTEM_PROMPT = """\
你是 wiki 质量评估专家。你的任务是判断 wiki 知识库是否需要补充或修复。

## 判定原则

- 不要因为 Agent 调用了 raw_read 就直接判定需要修复。
- raw_read/grep_raw 只是风险信号，表示 wiki 可能不完整。
- 只有在明确发现以下情况时，才判定 needs_repair=true：
  1. wiki 检索结果不足以支持回答，Agent 主要依赖 raw_read 才能回答
  2. 回答引用了 wiki 中不存在的关键信息
  3. 回答与 wiki 现有内容矛盾
- 如果 wiki 检索结果已经足够支持回答，即使 Agent 额外调用了 raw_read 做验证或补充文本，也不应判定修复。

## 输出要求

1. 先分析证据字段：wiki_search_sufficient、used_raw_read、raw_read_required_for_answer、answer_contains_info_missing_from_wiki、answer_contradicts_wiki
2. 再根据证据给出最终判定：needs_repair、confidence、reason
3. reason 必须与 needs_repair 保持一致：
   - needs_repair=false 时 reason 不能说"需要修复/缺失/不完整"
   - needs_repair=true 时 reason 不能说"wiki 内容充足/无缺陷"
4. confidence 只表示对当前判定的置信度，不表示问题严重程度

完成分析后，调用 submit_evaluation 工具提交结果。"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorInput:
    user_message: str
    assistant_answer: str
    tool_traces: list[dict]
    wiki_reads: list[dict]
    raw_reads: list[dict]


@dataclass
class EvaluatorOutput:
    needs_repair: bool
    confidence: str
    reason: str
    raw: dict


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_evaluator_prompt(inp: EvaluatorInput) -> list[dict]:
    user_content = (
        f"## 用户提问\n{inp.user_message}\n\n"
        f"## Agent 回答\n{inp.assistant_answer}\n\n"
        f"## 工具调用痕迹\n```json\n{_fmt(inp.tool_traces)}\n```\n\n"
        f"## Wiki 读取记录\n```json\n{_fmt(inp.wiki_reads)}\n```\n\n"
        f"## 原始文件读取记录\n```json\n{_fmt(inp.raw_reads)}\n```\n\n"
        "请根据以上内容，先分析证据字段，再给出最终判定，然后调用 submit_evaluation 工具提交。"
    )
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _fmt(obj: list | dict) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)[:8000]


# ---------------------------------------------------------------------------
# Parsing & normalization
# ---------------------------------------------------------------------------

def _clean_tool_string(value: object) -> str:
    """Remove provider/tool-call markup fragments that sometimes leak into args."""
    if not isinstance(value, str):
        return ""
    cleaned = re.sub(r"\s*</parameter\b[^>\n]*(?:>|$).*$", "", value, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"\s*</?[a-zA-Z_][\w:-]*\b[^>]*>\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _normalize_confidence(value: object) -> str:
    cleaned = _clean_tool_string(value).lower()
    for allowed in ("high", "medium", "low"):
        if cleaned == allowed or cleaned.startswith(f"{allowed}\n") or cleaned.startswith(f"{allowed} "):
            return allowed
    return "low"


def _parse_tool_call(tc: ToolCallResult) -> EvaluatorOutput:
    args = tc.arguments
    normalized = {
        "wiki_search_sufficient": bool(args.get("wiki_search_sufficient", True)),
        "used_raw_read": bool(args.get("used_raw_read", False)),
        "raw_read_required_for_answer": bool(args.get("raw_read_required_for_answer", False)),
        "answer_contains_info_missing_from_wiki": bool(args.get("answer_contains_info_missing_from_wiki", False)),
        "answer_contradicts_wiki": bool(args.get("answer_contradicts_wiki", False)),
        "needs_repair": bool(args.get("needs_repair", False)),
        "confidence": _normalize_confidence(args.get("confidence", "low")),
        "reason": _clean_tool_string(args.get("reason", "")),
    }
    return EvaluatorOutput(
        needs_repair=normalized["needs_repair"],
        confidence=normalized["confidence"],
        reason=normalized["reason"],
        raw=normalized,
    )


# ---------------------------------------------------------------------------
# Post-processing consistency check
# ---------------------------------------------------------------------------

_REPAIR_KEYWORDS = re.compile(
    r"需要修复|需要补充|缺失|不完整|不足|过时|缺少|缺乏|应补充|应修复|"
    r"raw_read\s*才能|依赖.*原始文件|wiki.*不足|信息.*缺失",
    re.IGNORECASE,
)

_NO_REPAIR_KEYWORDS = re.compile(
    r"无需修复|不需要修复|内容充足|无明显缺陷|wiki.*足够|无缺陷|充分|"
    r"不需要补充|没有缺失|完整.*准确",
    re.IGNORECASE,
)


def check_consistency(output: EvaluatorOutput) -> str | None:
    """Return a conflict description if the output is internally inconsistent,
    or None if it passes."""
    reason = output.reason
    raw = output.raw

    if not output.needs_repair and _REPAIR_KEYWORDS.search(reason):
        return (
            f"needs_repair=false 但 reason 包含修复相关关键词: '{reason}'"
        )

    if output.needs_repair and _NO_REPAIR_KEYWORDS.search(reason):
        return (
            f"needs_repair=true 但 reason 表示无需修复: '{reason}'"
        )

    if not output.needs_repair:
        evidence_signals = sum([
            raw.get("raw_read_required_for_answer", False),
            raw.get("answer_contains_info_missing_from_wiki", False),
            raw.get("answer_contradicts_wiki", False),
        ])
        if evidence_signals >= 2:
            return (
                f"needs_repair=false 但多项证据指向需要修复 "
                f"(raw_read_required={raw.get('raw_read_required_for_answer')}, "
                f"missing_info={raw.get('answer_contains_info_missing_from_wiki')}, "
                f"contradicts={raw.get('answer_contradicts_wiki')})"
            )

    if output.needs_repair and output.confidence == "high":
        if raw.get("wiki_search_sufficient") and not raw.get("raw_read_required_for_answer"):
            return (
                "needs_repair=true + confidence=high 但 wiki 检索充足且 raw_read 非必需，"
                "证据不支持 high confidence"
            )

    return None


def apply_evidence_override(output: EvaluatorOutput) -> EvaluatorOutput:
    """Apply evidence-based overrides when the evidence clearly contradicts
    the model's final verdict. Returns a corrected copy."""
    raw = output.raw

    if output.needs_repair:
        if (raw.get("wiki_search_sufficient")
                and not raw.get("raw_read_required_for_answer")
                and not raw.get("answer_contains_info_missing_from_wiki")
                and not raw.get("answer_contradicts_wiki")):
            logger.info("Evidence override: all evidence says no repair needed, overriding to false")
            return EvaluatorOutput(
                needs_repair=False,
                confidence="medium",
                reason="证据显示 wiki 检索充足且无缺失信息，覆盖为不需要修复",
                raw={**raw, "needs_repair": False, "confidence": "medium",
                     "reason": "evidence override: no repair needed",
                     "_overridden": True},
            )

    if not output.needs_repair:
        strong_signals = (
            raw.get("raw_read_required_for_answer", False)
            and raw.get("answer_contains_info_missing_from_wiki", False)
        )
        if strong_signals or raw.get("answer_contradicts_wiki", False):
            logger.info("Evidence override: strong evidence says repair needed, overriding to true")
            return EvaluatorOutput(
                needs_repair=True,
                confidence="medium",
                reason="证据显示回答依赖原始文件且包含 wiki 缺失信息，覆盖为需要修复",
                raw={**raw, "needs_repair": True, "confidence": "medium",
                     "reason": "evidence override: repair needed",
                     "_overridden": True},
            )

    return output


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_MAX_ATTEMPTS = 2


async def run_evaluator(
    inp: EvaluatorInput,
    cfg: FeedbackModelConfig,
) -> EvaluatorOutput:
    """Run the evaluator agent with consistency checks and auto-retry."""
    messages = _build_evaluator_prompt(inp)
    last_output: EvaluatorOutput | None = None

    for attempt in range(_MAX_ATTEMPTS):
        resp = await complete_with_tools(
            messages=messages,
            tools=[SUBMIT_EVALUATION_TOOL],
            cfg=cfg,
            tool_choice="auto",
        )

        parsed: EvaluatorOutput | None = None
        for tc in resp.tool_calls:
            if tc.name == "submit_evaluation":
                parsed = _parse_tool_call(tc)
                break

        if parsed is None:
            logger.warning("Evaluator attempt %d/%d: no tool call", attempt + 1, _MAX_ATTEMPTS)
            if resp.content:
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({
                    "role": "user",
                    "content": "请调用 submit_evaluation 工具提交你的评估结果。",
                })
            continue

        conflict = check_consistency(parsed)
        if conflict is None:
            return apply_evidence_override(parsed)

        logger.warning(
            "Evaluator attempt %d/%d: consistency conflict: %s",
            attempt + 1, _MAX_ATTEMPTS, conflict,
        )
        last_output = parsed

        if attempt < _MAX_ATTEMPTS - 1:
            messages.append({"role": "assistant", "content": resp.content or ""})
            tc_for_msg = resp.tool_calls[0]
            import json as _json
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tc_for_msg.id,
                    "type": "function",
                    "function": {
                        "name": tc_for_msg.name,
                        "arguments": _json.dumps(tc_for_msg.arguments, ensure_ascii=False),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc_for_msg.id,
                "content": f"[CONFLICT] {conflict}",
            })
            messages.append({
                "role": "user",
                "content": (
                    f"你的上一次评估存在不一致：{conflict}\n"
                    "请重新分析证据字段，确保 reason 与 needs_repair 一致，"
                    "然后重新调用 submit_evaluation 工具提交。"
                ),
            })

    if last_output:
        logger.info("Evaluator applying evidence override after conflict")
        return apply_evidence_override(last_output)

    return EvaluatorOutput(
        needs_repair=False,
        confidence="low",
        reason="evaluator did not produce structured output",
        raw={},
    )
