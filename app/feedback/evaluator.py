"""Feedback evaluator agent — uses tool calling for structured output.

Analyzes an agent conversation (messages + tool traces) to determine if the
wiki response has quality issues that warrant a repair.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import FeedbackModelConfig
from app.feedback.llm import ToolCallResult, complete_with_tools

logger = logging.getLogger(__name__)

SUBMIT_EVALUATION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_evaluation",
        "description": "提交 wiki 质量评估结果。分析完成后必须调用此工具提交判定。",
        "parameters": {
            "type": "object",
            "properties": {
                "needs_repair": {
                    "type": "boolean",
                    "description": "是否需要修复。",
                },
                "target_page_path": {
                    "type": "string",
                    "description": "建议修复的目标 wiki 页面路径（如不适用则留空字符串）。",
                },
                "page_exists": {
                    "type": "boolean",
                    "description": "该页面当前是否已存在于 wiki 中。",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "判定置信度。",
                },
                "reason": {
                    "type": "string",
                    "description": "一句话判定原因。",
                },
                "missing_info": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "wiki 中缺失或过时的信息列表。",
                },
                "suggested_sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "建议新增或修改的章节名称。",
                },
            },
            "required": ["needs_repair", "confidence", "reason"],
        },
    },
}

EVALUATOR_SYSTEM_PROMPT = """\
你是一个 wiki 质量评估专家。你的任务是分析 Agent 的对话记录（用户提问、Agent 回答、Agent 调用的工具痕迹），
判断 wiki 知识库是否存在缺陷（信息缺失、过时、不完整、自相矛盾等），是否需要生成修复候选。

评估规则：
1. 如果 Agent 使用了 raw_read（原始文件读取）来回答 wiki 相关问题，说明 wiki 内容可能不足，应判定需要修复。
2. 如果 Agent 的回答与 wiki 页面内容不一致或引用了 wiki 中不存在的信息，应判定需要修复。
3. 仅当明确发现 wiki 缺陷时才判定 needs_repair=true，避免误报。
4. 置信度要求：high 表示确定需要修复，medium 表示很可能需要修复，low 表示有一定可能。

完成分析后，必须调用 submit_evaluation 工具提交结果。不要输出 JSON，只通过工具调用提交。"""


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
    target_page_path: str
    page_exists: bool
    confidence: str
    reason: str
    missing_info: list[str]
    suggested_sections: list[str]
    raw: dict


def _build_evaluator_prompt(inp: EvaluatorInput) -> list[dict]:
    user_content = (
        f"## 用户提问\n{inp.user_message}\n\n"
        f"## Agent 回答\n{inp.assistant_answer}\n\n"
        f"## 工具调用痕迹\n```json\n{_fmt(inp.tool_traces)}\n```\n\n"
        f"## Wiki 读取记录\n```json\n{_fmt(inp.wiki_reads)}\n```\n\n"
        f"## 原始文件读取记录\n```json\n{_fmt(inp.raw_reads)}\n```\n\n"
        "请分析以上内容，判断 wiki 是否存在缺陷，然后调用 submit_evaluation 工具提交结果。"
    )
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _fmt(obj: list | dict) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)[:8000]


def _parse_tool_call(tc: ToolCallResult) -> EvaluatorOutput:
    args = tc.arguments
    return EvaluatorOutput(
        needs_repair=args.get("needs_repair", False),
        target_page_path=args.get("target_page_path", ""),
        page_exists=args.get("page_exists", True),
        confidence=args.get("confidence", "low"),
        reason=args.get("reason", ""),
        missing_info=args.get("missing_info", []),
        suggested_sections=args.get("suggested_sections", []),
        raw=args,
    )


async def run_evaluator(
    inp: EvaluatorInput,
    cfg: FeedbackModelConfig,
) -> EvaluatorOutput:
    """Run the evaluator agent. Returns parsed evaluation result."""
    messages = _build_evaluator_prompt(inp)
    resp = await complete_with_tools(
        messages=messages,
        tools=[SUBMIT_EVALUATION_TOOL],
        cfg=cfg,
    )

    for tc in resp.tool_calls:
        if tc.name == "submit_evaluation":
            return _parse_tool_call(tc)

    logger.warning("Evaluator did not call submit_evaluation tool, defaulting to no-repair")
    return EvaluatorOutput(
        needs_repair=False,
        target_page_path="",
        page_exists=True,
        confidence="low",
        reason="evaluator did not produce structured output",
        missing_info=[],
        suggested_sections=[],
        raw={},
    )
