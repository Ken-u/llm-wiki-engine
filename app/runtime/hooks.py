"""Startup hook execution for runtime integrations."""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from app.runtime.config import RuntimeSettings, get_runtime_config_path

logger = logging.getLogger("app.runtime.hooks")


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
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            return Path(bundle_root)
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


async def _run_command(
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> HookRunResult:
    """Execute a single hook command and log inputs/outputs for debugging."""
    start = time.monotonic()
    logger.info("Starting hook '%s' with command: %s", name, " ".join(shlex.quote(str(c)) for c in command))
    logger.debug("Hook '%s' cwd=%s env_keys=%s", name, cwd, sorted(env.keys()))

    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:
        logger.error("Hook '%s' failed to start process: %s", name, exc)
        return HookRunResult(
            name=name,
            status="failed",
            error=f"Failed to start process: {exc}",
        )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.error("Hook '%s' timed out after %d seconds", name, timeout_seconds)
        if proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        return HookRunResult(
            name=name,
            status="failed",
            elapsed_seconds=round(time.monotonic() - start, 3),
            error=f"Hook timed out after {timeout_seconds} seconds",
        )
    except Exception as exc:
        logger.error("Hook '%s' process communication failed: %s", name, exc)
        return HookRunResult(
            name=name,
            status="failed",
            elapsed_seconds=round(time.monotonic() - start, 3),
            error=f"Process communication failed: {exc}",
        )

    stdout_text = stdout.decode(errors="replace")[-4000:]
    stderr_text = stderr.decode(errors="replace")[-4000:]
    elapsed = round(time.monotonic() - start, 3)

    if stdout_text.strip():
        logger.info("Hook '%s' stdout (exit %d, %.3fs):\n%s", name, proc.returncode or 0, elapsed, stdout_text)
    if stderr_text.strip():
        logger.warning("Hook '%s' stderr (exit %d, %.3fs):\n%s", name, proc.returncode or 0, elapsed, stderr_text)

    result = HookRunResult(
        name=name,
        status="ok" if proc.returncode == 0 else "failed",
        exit_code=proc.returncode,
        elapsed_seconds=elapsed,
        stdout=stdout_text,
        stderr=stderr_text,
    )
    if proc.returncode != 0:
        result.error = f"Hook exited with code {proc.returncode}"
        logger.error("Hook '%s' failed with exit code %d after %.3fs", name, proc.returncode or -1, elapsed)
    else:
        logger.info("Hook '%s' completed successfully after %.3fs", name, elapsed)

    return result


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
        if not command:
            result = HookRunResult(
                name=script.name,
                status="skipped",
                error=f"No command configured for {platform_key}",
            )
            logger.warning("Hook '%s' skipped: no command configured for platform '%s'", script.name, platform_key)
            _last_results.append(result)
            if cfg.stop_on_failure:
                raise RuntimeError(result.error)
            continue

        result = await _run_command(
            script.name,
            command,
            cwd,
            env,
            cfg.timeout_seconds,
        )

        if result.status == "ok" and script.wait_for and script.wait_for.url:
            try:
                await _wait_for_url(script.wait_for.url, script.wait_for.timeout_seconds)
            except Exception as exc:
                result.status = "failed"
                result.error = str(exc)

        _last_results.append(result)
        if result.status == "failed" and cfg.stop_on_failure:
            logger.error("Startup hook '%s' failed and stop_on_failure is enabled; raising RuntimeError", result.name)
            raise RuntimeError(f"Startup hook failed: {result.name}: {result.error}")

    logger.info("All startup hooks completed; results=%s", [r.name + ":" + r.status for r in _last_results])
    return _last_results


def get_hook_results() -> list[dict]:
    return [asdict(r) for r in _last_results]
