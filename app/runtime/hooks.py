"""Startup hook execution for runtime integrations."""

from __future__ import annotations

import asyncio
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from app.runtime.config import HookRepositoryConfig, RuntimeSettings, get_runtime_config_path


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


def _inject_auth(url: str, token: str, *, username: str = "") -> str:
    if not token or not url.startswith(("http://", "https://")):
        return url
    parts = urlsplit(url)
    if "@" in parts.netloc:
        return url
    userinfo = quote(username or "token", safe="")
    password = quote(token, safe="")
    return urlunsplit((parts.scheme, f"{userinfo}:{password}@{parts.netloc}", parts.path, parts.query, parts.fragment))


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit((parts.scheme, f"***@{parts.netloc.split('@', 1)[1]}", parts.path, parts.query, parts.fragment))


def _run_git(args: list[str], *, cwd: Path | None = None, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _sync_repository(repo: HookRepositoryConfig, *, timeout_seconds: int) -> HookRunResult:
    start = time.monotonic()
    name = f"repository:{repo.name}"
    target = Path(repo.path)
    branch = repo.branch or "main"
    authed_url = _inject_auth(repo.url, repo.token, username=repo.username)
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def finish(status: str, *, exit_code: int | None = None, error: str = "") -> HookRunResult:
        return HookRunResult(
            name=name,
            status=status,
            exit_code=exit_code,
            elapsed_seconds=round(time.monotonic() - start, 3),
            stdout="\n".join(stdout_parts)[-4000:],
            stderr="\n".join(stderr_parts)[-4000:],
            error=error,
        )

    if not repo.enabled:
        return finish("skipped")
    if not repo.url:
        return finish("failed", error="Repository url is empty")

    try:
        if not (target / ".git").exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            result = _run_git(
                ["clone", "--branch", branch, "--single-branch", authed_url, str(target)],
                timeout_seconds=timeout_seconds,
            )
        else:
            result = _run_git(["remote", "set-url", "origin", authed_url], cwd=target, timeout_seconds=timeout_seconds)
            stdout_parts.append(result.stdout)
            stderr_parts.append(result.stderr)
            if result.returncode != 0:
                return finish("failed", exit_code=result.returncode, error="git remote set-url failed")

            result = _run_git(["fetch", "origin", branch], cwd=target, timeout_seconds=timeout_seconds)
            stdout_parts.append(result.stdout)
            stderr_parts.append(result.stderr)
            if result.returncode != 0:
                return finish("failed", exit_code=result.returncode, error="git fetch failed")

            checkout = _run_git(["checkout", branch], cwd=target, timeout_seconds=timeout_seconds)
            stdout_parts.append(checkout.stdout)
            stderr_parts.append(checkout.stderr)
            if checkout.returncode != 0:
                checkout = _run_git(["checkout", "-B", branch, f"origin/{branch}"], cwd=target, timeout_seconds=timeout_seconds)
                stdout_parts.append(checkout.stdout)
                stderr_parts.append(checkout.stderr)
                if checkout.returncode != 0:
                    return finish("failed", exit_code=checkout.returncode, error="git checkout failed")

            result = _run_git(["pull", "--ff-only", "origin", branch], cwd=target, timeout_seconds=timeout_seconds)

        stdout_parts.append(result.stdout)
        stderr_parts.append(result.stderr)
        if result.returncode != 0:
            return finish(
                "failed",
                exit_code=result.returncode,
                error=f"git sync failed for {_redact_url(repo.url)}",
            )
        return finish("ok", exit_code=0)
    except Exception as exc:
        return finish("failed", error=str(exc))


async def run_repository_sync_hooks(settings: RuntimeSettings) -> list[HookRunResult]:
    cfg = settings.hooks
    results: list[HookRunResult] = []
    for repo in cfg.repositories:
        result = await asyncio.to_thread(_sync_repository, repo, timeout_seconds=cfg.timeout_seconds)
        results.append(result)
        if result.status == "failed" and cfg.stop_on_failure:
            raise RuntimeError(f"Repository hook failed: {repo.name}: {result.error}")
    return results


async def run_startup_hooks(settings: RuntimeSettings) -> list[HookRunResult]:
    global _last_results
    cfg = settings.hooks
    _last_results = []
    if not cfg.enabled or not cfg.run_on_startup:
        return _last_results

    _last_results.extend(await run_repository_sync_hooks(settings))

    cwd = (get_runtime_config_path() or Path("config.yaml").resolve()).parent
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
