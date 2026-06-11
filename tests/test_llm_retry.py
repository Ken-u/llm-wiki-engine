"""Tests for LLM transient error retry behavior."""

import asyncio
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
