"""Repair compiler agent — agentic loop with wiki filesystem tools.

The compiler operates on a *working copy* of the wiki directory:
  1. Creates a temp snapshot of the wiki
  2. Runs a multi-turn LLM loop where the agent can Read/Glob/Edit/Grep/Create
  3. Collects all file changes as the repair candidate
  4. Cleans up the working copy

The original wiki is never modified until the user approves and applies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import FeedbackModelConfig
from app.feedback.llm import complete_with_tools
from app.feedback.wiki_tools import (
    ALL_TOOLS,
    FileChange,
    WorkingCopy,
    execute_tool,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 100

COMPILER_SYSTEM_PROMPT = """\
你是一个 wiki 修复专家。你拥有一组工具来读取和编辑 wiki 文件系统。

## 可用工具

- **read_file(path)** — 读取 wiki 文件内容
- **list_files(pattern)** — 列出匹配 glob 模式的文件
- **grep(pattern, path?)** — 在 wiki 中搜索内容
- **edit_file(path, old_string, new_string)** — 编辑文件（精确替换）
- **create_file(path, content)** — 创建新文件
- **submit_changes(summary, confidence)** — 完成后提交所有变更

## 工作流程

1. **了解仓库结构**：先用 list_files("*") 查看 wiki 根目录，再用 list_files("**/*.md") 查看全部页面
2. **阅读关键文件**：读取 index.md 了解页面索引结构，读取与问题相关的现有页面
3. **确定修复方案**：根据仓库现有结构，决定是修改现有子页面还是创建新的子页面
4. **执行修复**：使用 edit_file 修改子页面，或使用 create_file 创建新子页面
5. **更新索引**：如果创建了新页面，在 index.md 中添加对应链接
6. **提交变更**：调用 submit_changes 提交

## 仓库结构规范

wiki 仓库有明确的目录结构，你必须遵守：

- **index.md** — 页面索引，列出所有子页面的链接。**不要在此文件中写入正文内容**，只能添加/修改链接
- **overview.md** — 总览页面。**不要直接修改此文件**，除非审阅者明确指示
- **log.md** — 编译日志。**不要修改此文件**
- **sources/** — 子页面目录，所有具体内容页面都应放在这里
- **queries/** — 查询结果页面
- **comparisons/** — 对比分析页面
- **synthesis/** — 综合分析页面

## 修复原则

- **新增内容应创建独立子页面**（如 `sources/xxx.md`），不要把大段内容塞进根文档
- 修改现有子页面时，只改评估报告指出的缺陷，不要大幅重写无关内容
- 保持现有格式和风格一致
- 如果创建了新页面，记得在 index.md 中添加链接条目
- 如果存在审阅者指导，必须严格按照指导进行修改
- 修复内容应准确、完整，基于 Agent 回答和原始资料中的事实"""


@dataclass
class CompilerInput:
    user_message: str
    assistant_answer: str
    evaluator_result: dict
    review_guidance: str | None
    raw_reads: list[dict]
    wiki_dir: str


@dataclass
class CompilerOutput:
    changes: list[FileChange]
    change_summary: str
    confidence: str
    raw: dict


def _build_compiler_prompt(inp: CompilerInput) -> list[dict]:
    parts = []

    parts.append(
        f"## 评估结果\n置信度: {inp.evaluator_result.get('confidence', '?')}\n"
        f"原因: {inp.evaluator_result.get('reason', '未知')}\n"
    )
    parts.append(f"## 原始用户提问\n{inp.user_message}\n")
    parts.append(f"## Agent 回答\n{inp.assistant_answer[:8000]}\n")

    if inp.raw_reads:
        parts.append(
            f"## 原始文件读取记录\n```json\n{json.dumps(inp.raw_reads, ensure_ascii=False, indent=2)[:4000]}\n```\n"
        )

    if inp.review_guidance:
        parts.append(
            f"## 审阅者指导\n⚠️ 必须严格按照以下指导修改：\n{inp.review_guidance}\n"
        )

    parts.append(
        "\n请先用 list_files 了解 wiki 仓库结构，"
        "然后根据评估结果和 Agent 回答内容，确定需要修改或创建哪些子页面，"
        "最后调用 submit_changes 提交。"
    )

    return [
        {"role": "system", "content": COMPILER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


async def run_compiler(
    inp: CompilerInput,
    cfg: FeedbackModelConfig,
) -> CompilerOutput:
    """Run the compiler agent with an agentic tool-calling loop."""

    wc = WorkingCopy(original_wiki_dir=inp.wiki_dir)
    wc.create()

    try:
        return await _agent_loop(inp, cfg, wc)
    finally:
        wc.cleanup()


async def _agent_loop(
    inp: CompilerInput,
    cfg: FeedbackModelConfig,
    wc: WorkingCopy,
) -> CompilerOutput:
    messages = _build_compiler_prompt(inp)

    for iteration in range(MAX_ITERATIONS):
        logger.info("Compiler iteration %d/%d", iteration + 1, MAX_ITERATIONS)

        resp = await complete_with_tools(
            messages=messages,
            tools=ALL_TOOLS,
            cfg=cfg,
            tool_choice="auto",
        )

        if not resp.tool_calls:
            if resp.content:
                logger.info("Compiler produced text without tool calls, prompting to submit")
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({
                    "role": "user",
                    "content": "请调用 submit_changes 工具提交你的修改。",
                })
                continue
            break

        assistant_msg: dict = {"role": "assistant", "content": resp.content or None}
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
            }
            for tc in resp.tool_calls
        ]
        messages.append(assistant_msg)

        submit_result = None
        for tc in resp.tool_calls:
            result_str = execute_tool(wc, tc.name, tc.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })
            logger.debug("Tool %s -> %s", tc.name, result_str[:200])

            if tc.name == "submit_changes":
                submit_result = tc.arguments

        if submit_result is not None:
            changes = wc.collect_changes()
            return CompilerOutput(
                changes=changes,
                change_summary=submit_result.get("summary", ""),
                confidence=submit_result.get("confidence", "low"),
                raw=submit_result,
            )

    changes = wc.collect_changes()
    if changes:
        logger.warning("Compiler did not call submit_changes, but %d files changed", len(changes))
        return CompilerOutput(
            changes=changes,
            change_summary="(auto-collected: agent did not explicitly submit)",
            confidence="low",
            raw={},
        )

    raise RuntimeError("Compiler did not produce any wiki changes")
