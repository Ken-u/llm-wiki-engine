"""Resolve the externally reachable base URL for generated links."""

from __future__ import annotations

from fastapi import Request


def _first_header_value(value: str | None) -> str:
    if not value:
        return ""
    return value.split(",", 1)[0].strip()


def _append_forwarded_port(host: str, request: Request) -> str:
    if not host or ":" in host or host.startswith("["):
        return host
    port = _first_header_value(request.headers.get("x-forwarded-port"))
    if not port:
        return host
    if port in ("80", "443"):
        return host
    return f"{host}:{port}"


def external_base_url(request: Request) -> str:
    """Build ``scheme://host[:port]`` as seen by the client."""
    configured = _first_header_value(request.headers.get("x-public-base-url"))
    if configured:
        return configured.rstrip("/")

    forwarded_host = _append_forwarded_port(
        _first_header_value(request.headers.get("x-forwarded-host")),
        request,
    )
    if forwarded_host:
        scheme = _first_header_value(request.headers.get("x-forwarded-proto")) or request.url.scheme or "http"
        return f"{scheme}://{forwarded_host}".rstrip("/")

    host = _first_header_value(request.headers.get("host"))
    if host:
        scheme = request.url.scheme or "http"
        return f"{scheme}://{host}".rstrip("/")

    return str(request.base_url).rstrip("/")
