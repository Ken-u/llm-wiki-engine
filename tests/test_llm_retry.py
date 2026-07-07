"""Tests for LLM transient error retry behavior."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.llm import client


def test_complete_retries_transient_upstream_error():
    calls = 0

    async def flaky_completion(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("litellm.InternalServerError: upstream error")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="ok")),
            ],
        )

    async def run():
        litellm = SimpleNamespace(acompletion=flaky_completion)
        with patch.object(client, "_litellm", return_value=litellm):
            with patch("app.llm.client.asyncio.sleep", AsyncMock()):
                result = await client.complete("system", "user")
        assert result == "ok"
        assert calls == 2

    asyncio.run(run())


def test_complete_does_not_retry_non_transient_error():
    async def invalid_request(**_kwargs):
        raise RuntimeError("BadRequestError: invalid request")

    async def run():
        litellm = SimpleNamespace(acompletion=invalid_request)
        with patch.object(client, "_litellm", return_value=litellm):
            with patch("app.llm.client.asyncio.sleep", AsyncMock()) as sleep:
                try:
                    await client.complete("system", "user")
                except RuntimeError as exc:
                    assert "BadRequestError" in str(exc)
                else:
                    raise AssertionError("non-transient errors should not be retried")
        sleep.assert_not_awaited()

    asyncio.run(run())


def test_complete_passes_configured_timeout_to_litellm():
    captured_kwargs = {}

    async def completion(**kwargs):
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="ok")),
            ],
        )

    async def run():
        cfg = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="test-model",
                api_key="test-key",
                api_base=None,
                ingest_temperature=0.1,
                timeout=300,
            )
        )
        litellm = SimpleNamespace(acompletion=completion)
        with patch.object(client, "get_config", return_value=cfg):
            with patch.object(client, "_litellm", return_value=litellm):
                result = await client.complete("system", "user")
        assert result == "ok"
        assert captured_kwargs["timeout"] == 300

    asyncio.run(run())


def test_complete_with_tools_writes_debug_log_file_without_api_key(tmp_path):
    log_dir = tmp_path / "llm-debug"

    async def completion(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="search_wiki",
                                    arguments='{"query":"EDLA"}',
                                ),
                            )
                        ],
                    ),
                ),
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15),
        )

    async def run():
        cfg = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="test-model",
                api_key="secret-key",
                api_base=None,
                debug_llm_log=str(log_dir),
                chat_temperature=0.7,
                timeout=300,
            )
        )
        litellm = SimpleNamespace(acompletion=completion)
        with patch.object(client, "get_config", return_value=cfg):
            with patch.object(client, "_litellm", return_value=litellm):
                result = await client.complete_with_tools(
                    [{"role": "user", "content": "查 EDLA"}],
                    [{"type": "function", "function": {"name": "search_wiki"}}],
                )
        assert result.tool_calls[0].name == "search_wiki"

    asyncio.run(run())

    files = list(log_dir.glob("*.json"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8")
    data = json.loads(content)
    assert data["created_at"]
    assert data["tools"][0]["function"]["name"] == "search_wiki"
    assert data["llm"]["kind"] == "tool_completion"
    assert data["llm"]["tools_ref"] == "tools"
    assert "tools" not in data["llm"]["request"]
    assert data["llm"]["request"]["messages"][0]["content"] == "查 EDLA"
    assert data["llm"]["response"]["tool_calls"][0]["arguments"] == '{"query":"EDLA"}'
    assert "secret-key" not in content


def test_debug_log_context_groups_events_in_one_file(tmp_path):
    log_dir = tmp_path / "llm-debug"
    cfg = SimpleNamespace(llm=SimpleNamespace(debug_llm_log=str(log_dir)))

    with patch.object(client, "get_config", return_value=cfg):
        client.reset_debug_llm_log_context()
        client.log_debug_llm_event("tool_execution", {"name": "search_wiki", "result": {"ok": True}})
        client.log_debug_llm_event("tool_execution", {"name": "read_wiki", "result": {"ok": True}})
        client.finish_debug_llm_log_context()

    files = list(log_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert [event["name"] for event in data["tool_executions"]] == ["search_wiki", "read_wiki"]


def test_debug_log_context_overwrites_latest_llm_snapshot(tmp_path):
    log_dir = tmp_path / "llm-debug"
    cfg = SimpleNamespace(
        llm=SimpleNamespace(
            provider="openai",
            model="test-model",
            api_key="secret-key",
            api_base=None,
            debug_llm_log=str(log_dir),
            chat_temperature=0.7,
            timeout=300,
        )
    )
    calls = 0

    async def completion(**_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=f"ok-{calls}", tool_calls=[]),
                ),
            ],
            usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15),
        )

    async def run():
        litellm = SimpleNamespace(acompletion=completion)
        with patch.object(client, "get_config", return_value=cfg):
            with patch.object(client, "_litellm", return_value=litellm):
                client.reset_debug_llm_log_context()
                await client.complete_with_tools(
                    [{"role": "user", "content": "first"}],
                    [{"type": "function", "function": {"name": "search_wiki"}}],
                )
                await client.complete_with_tools(
                    [{"role": "user", "content": "second"}],
                    [{"type": "function", "function": {"name": "search_wiki"}}],
                )
                client.finish_debug_llm_log_context()

    asyncio.run(run())

    files = list(log_dir.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["llm_call_count"] == 2
    assert data["tools"][0]["function"]["name"] == "search_wiki"
    assert data["llm"]["request"]["messages"][0]["content"] == "second"
    assert data["llm"]["response"]["content"] == "ok-2"
    assert "tools" not in data["llm"]["request"]


def test_debug_log_without_context_writes_one_file_per_event(tmp_path):
    log_dir = tmp_path / "llm-debug"
    cfg = SimpleNamespace(llm=SimpleNamespace(debug_llm_log=str(log_dir)))

    with patch.object(client, "get_config", return_value=cfg):
        client.log_debug_llm_event("tool_execution", {"name": "first"})
        client.log_debug_llm_event("tool_execution", {"name": "second"})

    files = list(log_dir.glob("*.json"))
    assert len(files) == 2


def test_complete_with_tools_can_be_cancelled_by_disconnect_signal():
    async def very_slow_completion(**_kwargs):
        await asyncio.sleep(10)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="late", tool_calls=[]),
                ),
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def should_cancel() -> bool:
        return True

    async def run():
        cfg = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="test-model",
                api_key="secret-key",
                api_base=None,
                debug_llm_log="",
                chat_temperature=0.7,
                timeout=300,
            )
        )
        litellm = SimpleNamespace(acompletion=very_slow_completion)
        with patch.object(client, "get_config", return_value=cfg):
            with patch.object(client, "_litellm", return_value=litellm):
                try:
                    await client.complete_with_tools(
                        [{"role": "user", "content": "cancel"}],
                        [{"type": "function", "function": {"name": "search_wiki"}}],
                        should_cancel=should_cancel,
                    )
                except asyncio.CancelledError:
                    return
                raise AssertionError("Expected cancellation when client disconnects")

    asyncio.run(run())


def test_stream_can_be_cancelled_while_waiting_next_chunk():
    class SlowStream:
        async def __anext__(self):
            await asyncio.sleep(10)
            raise StopAsyncIteration

        async def aclose(self):
            return None

    async def stream_completion(**_kwargs):
        return SlowStream()

    async def should_cancel() -> bool:
        return True

    async def run():
        cfg = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="test-model",
                api_key="secret-key",
                api_base=None,
                debug_llm_log="",
                chat_temperature=0.7,
                timeout=300,
            )
        )
        litellm = SimpleNamespace(acompletion=stream_completion)
        with patch.object(client, "get_config", return_value=cfg):
            with patch.object(client, "_litellm", return_value=litellm):
                stream_gen = client.stream(
                    [{"role": "user", "content": "cancel stream"}],
                    should_cancel=should_cancel,
                )
                try:
                    await stream_gen.__anext__()
                except asyncio.CancelledError:
                    return
                raise AssertionError("Expected streaming cancellation when client disconnects")

    asyncio.run(run())


def test_complete_with_tools_stream_reconstructs_tool_calls():
    class ToolCallStream:
        def __init__(self):
            self._chunks = [
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call-1",
                                        function=SimpleNamespace(name="search_wiki", arguments='{"query":"ED'),
                                    ),
                                ],
                            ),
                        )
                    ],
                    usage=None,
                ),
                SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="tool_calls",
                            delta=SimpleNamespace(
                                content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id=None,
                                        function=SimpleNamespace(name=None, arguments='LA"}'),
                                    ),
                                ],
                            ),
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
                ),
            ]
            self._idx = 0

        async def __anext__(self):
            if self._idx >= len(self._chunks):
                raise StopAsyncIteration
            chunk = self._chunks[self._idx]
            self._idx += 1
            return chunk

        async def aclose(self):
            return None

    async def stream_completion(**_kwargs):
        return ToolCallStream()

    async def run():
        cfg = SimpleNamespace(
            llm=SimpleNamespace(
                provider="openai",
                model="test-model",
                api_key="secret-key",
                api_base=None,
                debug_llm_log="",
                chat_temperature=0.7,
                timeout=300,
                stream=True,
            )
        )
        litellm = SimpleNamespace(acompletion=stream_completion)
        with patch.object(client, "get_config", return_value=cfg):
            with patch.object(client, "_litellm", return_value=litellm):
                result = await client.complete_with_tools(
                    [{"role": "user", "content": "查 EDLA"}],
                    [{"type": "function", "function": {"name": "search_wiki"}}],
                )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call-1"
        assert result.tool_calls[0].name == "search_wiki"
        assert result.tool_calls[0].arguments == '{"query":"EDLA"}'
        assert result.finish_reason == "tool_calls"
        assert result.usage is not None and result.usage.prompt_tokens == 10

    asyncio.run(run())
