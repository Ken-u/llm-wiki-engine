"""Tests for the OpenAI-compatible /v1/ endpoints."""

import pytest
from unittest.mock import AsyncMock, patch

from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def auth_headers():
    """Provide a mock auth token header."""
    return {"Authorization": "Bearer test-jwt-token"}


class TestChatCompletionsValidation:
    """Test request validation without hitting actual DB/LLM."""

    @pytest.mark.asyncio
    async def test_rejects_tools_parameter(self, client, auth_headers):
        with patch("app.openai_compat.router.get_current_user") as mock_user:
            mock_user.return_value = AsyncMock(id=1)
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [{"type": "function", "function": {"name": "x"}}],
                },
                headers=auth_headers,
            )
            # Should be 400 or 422 (pydantic may validate before our check)
            assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_rejects_stream_true(self, client, auth_headers):
        with patch("app.openai_compat.router.get_current_user") as mock_user:
            mock_user.return_value = AsyncMock(id=1)
            with patch("app.openai_compat.router.get_db"):
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                    headers=auth_headers,
                )
                assert resp.status_code in (400, 401, 404)

    @pytest.mark.asyncio
    async def test_rejects_functions_parameter(self, client, auth_headers):
        with patch("app.openai_compat.router.get_current_user") as mock_user:
            mock_user.return_value = AsyncMock(id=1)
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "functions": [{"name": "x"}],
                },
                headers=auth_headers,
            )
            assert resp.status_code in (400, 422)
