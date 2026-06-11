"""Agent orchestrator: tool-calling loop between LLM and agent tools.

Handles:
1. System prompt construction with tool use policy
2. Registering tool schemas with LLM
3. Executing tool calls and feeding results back
4. Per-call budget tracking — rejects excess calls via tool result
5. Streaming final text response via SSE events
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import AsyncGenerator

from app.agents.tools import ToolContext, execute_tool, get_tool_definitions
from app.config import get_config
from app.llm.client import complete_with_tools, LLMResponse, stream as llm_stream

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_CALLS = 20

REJECT_MSG = "已达到最大工具调用次数，请立即根据目前已获取的数据输出结果。"


@dataclass
class ToolTrace:
    """Record of a single tool call for debugging."""
    name: str
    arguments: dict
    result: dict


def _build_system_prompt(custom_prompt: str, has_ticket: bool) -> str:
    """Build system prompt with tool use policy."""
    ticket_policy = ""
    if has_ticket:
        ticket_policy = "\n".join([
            "",
            "案例库使用策略：",
            "- 如果用户问题涉及历史案例、具体故障经验、类似 issue、处理先例，可调用 search_ticket_cases 搜索案例库。",
            "- 搜索案例后，优先基于候选摘要（problem_summary、root_cause、resolution）直接回答。",
            "- 只有需要具体排查步骤、日志、完整上下文时才调用 read_ticket_case。",
            "- 不要一次读取多个案例全文。",
            "- 使用案例库时，回答末尾列出 case_id + title。",
        ])

    policy = "\n".join([
        "你是一个知识问答助手。你可以使用工具来检索知识库和案例库。",
        "",
        "工具使用策略：",
        "- 优先使用 search_wiki 搜索与问题相关的知识页面。",
        "- 搜索结果中的 snippet 不足以支撑结论时，使用 read_wiki_page 读取页面全文。",
        "- 索引搜索不到时，可使用 grep_raw 直接在原文件中搜索关键词。",
        "- 不要猜测答案，始终基于工具返回的知识库内容回答。",
        f"{ticket_policy}",
        "- 引用信息时使用 [[page-name]] 格式。",
        "- 如果知识库中没有找到相关信息，请如实说明。",
    ])

    if custom_prompt:
        return f"{custom_prompt}\n\n{policy}"
    return policy


DEFAULT_DEBUG_RESULT_LIMIT = 2000


def _truncate_for_debug(result: dict, limit: int = DEFAULT_DEBUG_RESULT_LIMIT) -> dict:
    """Truncate long string values in tool results for SSE debug events."""
    out = {}
    for k, v in result.items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit] + f"\n... ({len(v)} chars total)"
        elif isinstance(v, list) and len(v) > 30:
            out[k] = v[:30]
            out[f"_{k}_truncated"] = f"{len(v)} items total, showing first 30"
        else:
            out[k] = v
    return out


def _done_event(traces: list[ToolTrace]) -> str:
    return json.dumps({
        "done": True,
        "tool_traces": [{"name": t.name, "arguments": t.arguments} for t in traces],
    })


async def run_agent_turn(
    message: str,
    history: list[dict],
    system_prompt: str,
    ctx: ToolContext,
    *,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    debug_result_limit: int = DEFAULT_DEBUG_RESULT_LIMIT,
) -> AsyncGenerator[str, None]:
    """Run one agent turn: tool-calling loop then stream final answer.

    Yields SSE-compatible JSON strings:
      {"token": "..."}          — streamed text chunk
      {"tool_call": {...}}      — tool call event (may include "rejected": true)
      {"done": true, ...}       — completion marker
    """
    full_system = _build_system_prompt(system_prompt, ctx.ticket_project is not None)
    tool_defs = get_tool_definitions(ctx)

    messages: list[dict] = [{"role": "system", "content": full_system}]
    messages.extend(history[-20:])
    messages.append({"role": "user", "content": message})

    traces: list[ToolTrace] = []
    used_ticket = False
    tool_call_count = 0

    # Upper bound on LLM rounds: max_tool_calls + 2 to allow rejection + final text
    for _round in range(max_tool_calls + 2):
        resp: LLMResponse = await complete_with_tools(
            messages, tool_defs, max_tokens=4096,
        )

        if not resp.tool_calls:
            if resp.content:
                resp_content = resp.content
                if used_ticket and "参考了案例库" not in resp.content:
                    resp_content += "\n\n> 以上结论参考了案例库。"
                yield json.dumps({"token": resp_content})
            yield _done_event(traces)
            return

        # Build assistant message
        assistant_msg: dict = {"role": "assistant", "content": resp.content, "tool_calls": []}
        for tc in resp.tool_calls:
            assistant_msg["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            })
        messages.append(assistant_msg)

        for tc in resp.tool_calls:
            try:
                args = json.loads(tc.arguments)
            except json.JSONDecodeError:
                args = {}

            if tool_call_count >= max_tool_calls:
                yield json.dumps({"tool_call": {"name": tc.name, "arguments": args, "rejected": True}})
                reject_result = {"rejected": True, "message": REJECT_MSG}
                yield json.dumps({"tool_result": {"name": tc.name, "result": reject_result}})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": REJECT_MSG,
                })
                traces.append(ToolTrace(name=tc.name, arguments=args, result=reject_result))
            else:
                yield json.dumps({"tool_call": {"name": tc.name, "arguments": args}})
                result = await execute_tool(tc.name, args, ctx)
                yield json.dumps({"tool_result": {"name": tc.name, "result": _truncate_for_debug(result, debug_result_limit)}})
                traces.append(ToolTrace(name=tc.name, arguments=args, result=result))

                if tc.name in ("search_ticket_cases", "read_ticket_case"):
                    if "error" not in result:
                        used_ticket = True

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
                tool_call_count += 1

    # Exhausted all rounds — hard fallback, stream without tools
    cfg = get_config()
    async for token in llm_stream(
        messages,
        temperature=cfg.llm.chat_temperature,
        max_tokens=4096,
    ):
        yield json.dumps({"token": token})

    yield _done_event(traces)
