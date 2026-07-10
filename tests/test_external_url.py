"""Tests for external base URL resolution."""

from types import SimpleNamespace

from app.http.external_url import external_base_url


def _request(
    *,
    base_url: str = "http://internal:8000/",
    scheme: str = "http",
    port: int | None = 8000,
    headers: dict[str, str] | None = None,
):
    return SimpleNamespace(
        base_url=base_url,
        url=SimpleNamespace(scheme=scheme, port=port),
        headers=headers or {},
    )


def test_external_base_url_uses_host_header_with_port():
    url = external_base_url(_request(headers={"host": "wiki.example.com:8080"}))
    assert url == "http://wiki.example.com:8080"


def test_external_base_url_uses_forwarded_host_and_proto():
    url = external_base_url(_request(headers={
        "x-forwarded-host": "wiki.example.com:8443",
        "x-forwarded-proto": "https",
    }))
    assert url == "https://wiki.example.com:8443"


def test_external_base_url_appends_forwarded_port_when_missing():
    url = external_base_url(_request(headers={
        "x-forwarded-host": "wiki.example.com",
        "x-forwarded-proto": "http",
        "x-forwarded-port": "8080",
    }))
    assert url == "http://wiki.example.com:8080"


def test_external_base_url_prefers_public_override_header():
    url = external_base_url(_request(headers={
        "x-public-base-url": "https://public.example.com:9443",
        "host": "wiki.example.com:8080",
    }))
    assert url == "https://public.example.com:9443"
