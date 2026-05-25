"""SSE streaming helpers for FastAPI."""

from __future__ import annotations

import json
from typing import AsyncGenerator

from starlette.responses import StreamingResponse


def sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    async def event_stream():
        async for data in generator:
            yield f"data: {json.dumps({'token': data})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
