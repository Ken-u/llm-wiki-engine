"""Startup hook execution for runtime integrations."""

from __future__ import annotations

import asyncio
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from app.runtime.config import RuntimeSettings, get_runtime_config_path


@dataclass
class HookRunResult:
    name: str
    status: str
    exit_code: int | None = None
    elapsed_seconds: float = 0
    stdout: str = ""
    stderr: str = ""
    error: str = ""


_last_results: list[HookRunResult] = []


def _platform_key() -> str:
    name = platform.system().lower()
    if name == "windows":
        return "windows"
    if name == "darwin":
        return "darwin"
    return "linux"


def _runtime_app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


async def _wait_for_url(url: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(url)
            if resp.status_code < 500:
                return
        except Exception as exc:
            last_error = exc
        await asyncio.sleep(1)
    if last_error:
        raise TimeoutError(f"Timed out waiting for {url}: {last_error}") from last_error
    raise TimeoutError(f"Timed out waiting for {url}")


async def run_startup_hooks(settings: RuntimeSettings) -> list[HookRunResult]:
    global _last_results
    cfg = settings.hooks
    _last_results = []
    if not cfg.enabled or not cfg.run_on_startup:
        return _last_results

    config_path = get_runtime_config_path() or Path("config.yaml").resolve()
    cwd = config_path.parent
    env = {
        **os.environ,
        "RUNTIME_CONFIG": str(config_path),
        "RUNTIME_CONFIG_DIR": str(cwd),
        "RUNTIME_APP_DIR": str(_runtime_app_dir()),
        "RUNTIME_PLATFORM": _platform_key(),
    }
    platform_key = _platform_key()

    for script in cfg.scripts:
        command = getattr(script.command, platform_key)
        start = time.monotonic()
        if not command:
            result = HookRunResult(
                name=script.name,
                status="skipped",
                error=f"No command configured for {platform_key}",
            )
            _last_results.append(result)
            if cfg.stop_on_failure:
                raise RuntimeError(result.error)
            continue

        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=cfg.timeout_seconds,
            )
            result = HookRunResult(
                name=script.name,
                status="ok" if proc.returncode == 0 else "failed",
                exit_code=proc.returncode,
                elapsed_seconds=round(time.monotonic() - start, 3),
                stdout=stdout.decode(errors="replace")[-4000:],
                stderr=stderr.decode(errors="replace")[-4000:],
            )
            if proc.returncode != 0:
                result.error = f"Hook exited with code {proc.returncode}"
        except Exception as exc:
            result = HookRunResult(
                name=script.name,
                status="failed",
                elapsed_seconds=round(time.monotonic() - start, 3),
                error=str(exc),
            )

        if result.status == "ok" and script.wait_for and script.wait_for.url:
            try:
                await _wait_for_url(script.wait_for.url, script.wait_for.timeout_seconds)
            except Exception as exc:
                result.status = "failed"
                result.error = str(exc)

        _last_results.append(result)
        if result.status == "failed" and cfg.stop_on_failure:
            raise RuntimeError(f"Startup hook failed: {result.name}: {result.error}")

    return _last_results


def get_hook_results() -> list[dict]:
    return [asdict(r) for r in _last_results]
