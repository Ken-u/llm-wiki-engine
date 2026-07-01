"""Standalone read-only runtime entry point.

Run with:
    uvicorn app.runtime_main:app --host 127.0.0.1 --port 8012

Or:
    python -m app.runtime_main --config ./config.yaml
"""

from __future__ import annotations

import argparse
import os
import socket
import webbrowser
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.runtime.config import get_runtime_config, load_runtime_config
from app.runtime.hooks import run_startup_hooks
from app.runtime.router import openai_router, router as runtime_router
from app.runtime.ui import mount_runtime_ui


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) != 0


def _choose_port(host: str, port: int) -> int:
    if _port_available(host, port):
        return port
    for candidate in range(port + 1, port + 50):
        if _port_available(host, candidate):
            return candidate
    return port


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_runtime_config()
    if not os.environ.get("RUNTIME_SKIP_HOOKS") and settings.hooks.run_before_server:
        await run_startup_hooks(settings)
    yield


app = FastAPI(
    title="LLM Wiki Runtime",
    description="Single-project local read-only inference runtime",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runtime_router)
app.include_router(openai_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


mount_runtime_ui(app)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM Wiki Runtime")
    parser.add_argument("--config", default=os.environ.get("RUNTIME_CONFIG", "config.yaml"))
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--skip-hooks", action="store_true")
    args = parser.parse_args()

    os.environ["RUNTIME_CONFIG"] = args.config
    if args.skip_hooks:
        os.environ["RUNTIME_SKIP_HOOKS"] = "1"

    settings = load_runtime_config(args.config)
    host = args.host or settings.server.host
    port = args.port or settings.server.port
    if args.port is None:
        port = _choose_port(host, port)

    if settings.server.open_browser and not args.no_browser:
        webbrowser.open(f"http://{host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
