#!/usr/bin/env python3
"""Clone or fast-forward repositories declared in runtime-config.yaml.

This hook intentionally lives outside the runtime application. The app only
executes configured hook commands and passes RUNTIME_CONFIG to them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit


def _clean_scalar(value: str) -> str:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def _parse_repositories(config_path: Path) -> list[dict[str, str]]:
    repos: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_repositories = False
    repositories_indent = 0

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue

        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        stripped = line_without_comment.strip()

        if stripped == "repositories:":
            in_repositories = True
            repositories_indent = indent
            continue

        if in_repositories and indent <= repositories_indent and not stripped.startswith("- "):
            break

        if not in_repositories:
            continue

        if stripped.startswith("- "):
            if current is not None:
                repos.append(current)
            current = {}
            rest = stripped[2:].strip()
            if rest and ":" in rest:
                key, value = rest.split(":", 1)
                current[key.strip()] = _clean_scalar(value)
            continue

        if current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current[key.strip()] = _clean_scalar(value)

    if current is not None:
        repos.append(current)
    return repos


def _resolve_path(config_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def _inject_auth(url: str, token: str, username: str = "") -> str:
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


def _redact(text: str, authed_url: str, token: str) -> str:
    text = text.replace(authed_url, _redact_url(authed_url))
    if token:
        text = text.replace(token, "***")
    return text


def _run_git(args: list[str], cwd: Path | None, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _sync_repo(repo: dict[str, str], config_dir: Path, timeout: int) -> dict:
    started = time.monotonic()
    name = repo.get("name") or repo.get("path") or "repository"
    enabled = repo.get("enabled", "true").lower() not in {"0", "false", "no", "off"}
    url = repo.get("url", "")
    branch = repo.get("branch", "").strip()
    token = repo.get("token", "")
    authed_url = _inject_auth(url, token, repo.get("username", ""))
    target = _resolve_path(config_dir, repo.get("path", ""))
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def finish(status: str, exit_code: int | None = None, error: str = "") -> dict:
        stdout = _redact("\n".join(stdout_parts), authed_url, token)
        stderr = _redact("\n".join(stderr_parts), authed_url, token)
        if status == "failed" and stderr and stderr not in error:
            error = f"{error}: {stderr[-1000:]}" if error else stderr[-1000:]
        return {
            "name": name,
            "status": status,
            "exit_code": exit_code,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "path": str(target),
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
            "error": error,
        }

    if not enabled:
        return finish("skipped")
    if not url:
        return finish("failed", error="url is required")
    if not repo.get("path"):
        return finish("failed", error="path is required")

    try:
        if not (target / ".git").exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            args = ["clone", authed_url, str(target)]
            if branch:
                args = ["clone", "--branch", branch, "--single-branch", authed_url, str(target)]
            result = _run_git(args, None, timeout)
        else:
            result = _run_git(["remote", "set-url", "origin", authed_url], target, timeout)
            stdout_parts.append(result.stdout)
            stderr_parts.append(result.stderr)
            if result.returncode != 0:
                return finish("failed", result.returncode, "git remote set-url failed")

            fetch_args = ["fetch", "origin", branch] if branch else ["fetch", "origin"]
            result = _run_git(fetch_args, target, timeout)
            stdout_parts.append(result.stdout)
            stderr_parts.append(result.stderr)
            if result.returncode != 0:
                return finish("failed", result.returncode, "git fetch failed")

            if branch:
                result = _run_git(["checkout", branch], target, timeout)
                stdout_parts.append(result.stdout)
                stderr_parts.append(result.stderr)
                if result.returncode != 0:
                    result = _run_git(["checkout", "-B", branch, f"origin/{branch}"], target, timeout)
                    stdout_parts.append(result.stdout)
                    stderr_parts.append(result.stderr)
                    if result.returncode != 0:
                        return finish("failed", result.returncode, "git checkout failed")

            pull_args = ["pull", "--ff-only", "origin", branch] if branch else ["pull", "--ff-only"]
            result = _run_git(pull_args, target, timeout)

        stdout_parts.append(result.stdout)
        stderr_parts.append(result.stderr)
        if result.returncode != 0:
            return finish("failed", result.returncode, f"git sync failed for {_redact_url(url)}")
        return finish("ok", 0)
    except Exception as exc:
        return finish("failed", error=str(exc))


def main() -> int:
    config_path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUNTIME_CONFIG", "runtime-config.yaml"))
    config_path = config_path.expanduser().resolve()
    timeout = int(os.environ.get("SYNC_REPOSITORIES_TIMEOUT", "120"))
    repos = _parse_repositories(config_path)
    results = [_sync_repo(repo, config_path.parent, timeout) for repo in repos]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if any(result["status"] == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
