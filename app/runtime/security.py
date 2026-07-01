"""Optional static API key support for runtime routes."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.runtime.config import get_runtime_config


async def require_runtime_api_key(authorization: str | None = Header(None)) -> None:
    api_key = get_runtime_config().server.api_key
    if not api_key:
        return
    raw = (authorization or "").replace("Bearer ", "").strip()
    if raw != api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")

