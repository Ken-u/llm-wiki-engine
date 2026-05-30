"""Repair compiler agent — uses tool calling for structured output.

Given evaluation results and optional reviewer guidance, generates a wiki
page repair candidate via LLM tool calling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import FeedbackModelConfig
from app.feedback.llm import ToolCallResult, complete_with_tools

logger = logging.getLogger(__name__)

SUBMIT_REPAIR_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_repair",
        "description": "提交单页修复候选。生成完成后必须调用此工具提交修复结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "proposed_content": {
                    "type": "string",
                    "description": "修复后的完整页面 markdown 内容。",
                },
                "change_summary": {
                    "type": "string",
                    "description": "简要描述本次修改了什么。",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "对本次修复质量的置信度。",
                },
                "sections_modified": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "修改涉及的章节标题列表。",
                },
            },
            "required": ["proposed_content", "change_summary", "confidence"],
        },
    },
}

COMPILER_SYSTEM_PROMPT = """\
你是一个 wiki 修复专家。你的任务是根据评估结果和原始对话，生成一份修复后的 wiki 页面内容。

修复规则：
1. 只修改评估报告中指出的缺陷，不要大幅重写无关内容。
2. 保持现有格式和风格一致。
3. 如果页面不存在（page_exists=false），创建一个完整的新页面。
4. 如果存在审阅者指导（review_guidance），必须严格按照指导进行修改。
5. 修复内容应准确、完整，可直接替换原页面。

完成修复后，必须调用 submit_repair 工具提交结果。不要输出 JSON，只通过工具调用提交。"""


@dataclass
class CompilerInput:
    user_message: str
    assistant_answer: str
    evaluator_result: dict
    target_page_path: str
    existing_page_content: str | None
    review_guidance: str | None
    wiki_reads: list[dict]
    raw_reads: list[dict]


@dataclass
class CompilerOutput:
    proposed_content: str
    change_summary: str
    confidence: str
    sections_modified: list[str]
    raw: dict


def _build_compiler_prompt(inp: CompilerInput) -> list[dict]:
    import json

    parts = [
        f"## 目标页面\n路径: `{inp.target_page_path}`\n",
    ]
    if inp.existing_page_content:
        parts.append(f"## 当前页面内容\n```markdown\n{inp.existing_page_content[:12000]}\n```\n")
    else:
        parts.append("## 当前页面内容\n（页面不存在，需新建）\n")

    parts.append(
        f"## 评估结果\n```json\n{json.dumps(inp.evaluator_result, ensure_ascii=False, indent=2)[:4000]}\n```\n"
    )
    parts.append(f"## 原始用户提问\n{inp.user_message}\n")
    parts.append(f"## Agent 回答\n{inp.assistant_answer[:6000]}\n")

    if inp.wiki_reads:
        parts.append(f"## Wiki 读取记录\n```json\n{json.dumps(inp.wiki_reads, ensure_ascii=False, indent=2)[:4000]}\n```\n")
    if inp.raw_reads:
        parts.append(f"## 原始文件读取记录\n```json\n{json.dumps(inp.raw_reads, ensure_ascii=False, indent=2)[:4000]}\n```\n")

    if inp.review_guidance:
        parts.append(f"## 审阅者指导\n⚠️ 必须严格按照以下指导修改：\n{inp.review_guidance}\n")

    parts.append("\n请根据以上信息生成修复后的完整页面内容，然后调用 submit_repair 工具提交。")

    return [
        {"role": "system", "content": COMPILER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def _parse_tool_call(tc: ToolCallResult) -> CompilerOutput:
    args = tc.arguments
    return CompilerOutput(
        proposed_content=args.get("proposed_content", ""),
        change_summary=args.get("change_summary", ""),
        confidence=args.get("confidence", "low"),
        sections_modified=args.get("sections_modified", []),
        raw=args,
    )


async def run_compiler(
    inp: CompilerInput,
    cfg: FeedbackModelConfig,
) -> CompilerOutput:
    """Run the compiler agent. Returns parsed repair candidate."""
    messages = _build_compiler_prompt(inp)
    resp = await complete_with_tools(
        messages=messages,
        tools=[SUBMIT_REPAIR_TOOL],
        cfg=cfg,
    )

    for tc in resp.tool_calls:
        if tc.name == "submit_repair":
            return _parse_tool_call(tc)

    logger.warning("Compiler did not call submit_repair tool")
    raise RuntimeError("Compiler did not produce structured repair output via tool call")
