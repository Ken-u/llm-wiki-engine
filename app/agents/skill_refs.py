"""Signed source references for installed Agent Skills."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from app.config import get_config


@dataclass(frozen=True)
class SourceRefPayload:
    agent_id: str
    project_id: str
    doc_name: str
    display_name: str
    exp: int


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def _secret() -> bytes:
    return get_config().auth.jwt_secret.encode("utf-8")


def _signature(payload: str) -> str:
    digest = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).digest()
    return _b64(digest)


def sign_source_ref(
    *,
    agent_id: str,
    project_id: str,
    doc_name: str,
    display_name: str,
    max_age_seconds: int = 3600,
) -> str:
    payload = {
        "agent_id": agent_id,
        "project_id": project_id,
        "doc_name": doc_name,
        "display_name": display_name,
        "exp": int(time.time()) + max_age_seconds,
    }
    encoded = _b64(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    return f"{encoded}.{_signature(encoded)}"


def verify_source_ref(ref: str, *, max_age_seconds: int = 3600) -> SourceRefPayload | None:
    try:
        encoded, signature = ref.split(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(_signature(encoded), signature):
        return None
    try:
        data = json.loads(_unb64(encoded).decode("utf-8"))
        payload = SourceRefPayload(
            agent_id=str(data["agent_id"]),
            project_id=str(data["project_id"]),
            doc_name=str(data["doc_name"]),
            display_name=str(data.get("display_name") or data["doc_name"]),
            exp=int(data["exp"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    now = int(time.time())
    if payload.exp < now:
        return None
    if payload.exp > now + max_age_seconds:
        return None
    return payload
