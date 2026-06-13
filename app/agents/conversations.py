"""File-backed persistent conversations for logged-in Agent users."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _root(base_dir: str | Path, agent_id: str, user_id: int) -> Path:
    return Path(base_dir) / "agent-chats" / agent_id / str(user_id)


def _list_path(base_dir: str | Path, agent_id: str, user_id: int) -> Path:
    return _root(base_dir, agent_id, user_id) / "conversations.json"


def _messages_path(base_dir: str | Path, agent_id: str, user_id: int, conversation_id: str) -> Path:
    return _root(base_dir, agent_id, user_id) / f"{conversation_id}.json"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_conversations(base_dir: str | Path, *, agent_id: str, user_id: int) -> list[dict]:
    convs = _load_json(_list_path(base_dir, agent_id, user_id), [])
    return sorted(convs, key=lambda c: c.get("updated_at") or c.get("created_at") or "", reverse=True)


def get_conversation(
    base_dir: str | Path,
    *,
    agent_id: str,
    user_id: int,
    conversation_id: str,
) -> dict | None:
    messages_file = _messages_path(base_dir, agent_id, user_id, conversation_id)
    if not messages_file.exists():
        return None
    return {
        "id": conversation_id,
        "messages": _load_json(messages_file, []),
    }


def append_turn(
    base_dir: str | Path,
    *,
    agent_id: str,
    user_id: int,
    conversation_id: str | None,
    user_message: str,
    assistant_answer: str,
    compressed_history: list[dict] | None = None,
) -> dict:
    conv_id = conversation_id or str(uuid.uuid4())
    messages_file = _messages_path(base_dir, agent_id, user_id, conv_id)
    if compressed_history is not None:
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in compressed_history
            if m.get("role") in ("user", "assistant")
        ]
    else:
        messages = _load_json(messages_file, [])
    messages.extend([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_answer, "rawContent": assistant_answer},
    ])
    _write_json(messages_file, messages)

    now = datetime.now(timezone.utc).isoformat()
    convs = _load_json(_list_path(base_dir, agent_id, user_id), [])
    existing = next((c for c in convs if c.get("id") == conv_id), None)
    if existing:
        existing["message_count"] = len(messages)
        existing["updated_at"] = now
        conv = existing
    else:
        conv = {
            "id": conv_id,
            "title": user_message[:80],
            "created_at": now,
            "updated_at": now,
            "message_count": len(messages),
        }
        convs.append(conv)
    _write_json(_list_path(base_dir, agent_id, user_id), convs)
    return conv


def delete_conversation(
    base_dir: str | Path,
    *,
    agent_id: str,
    user_id: int,
    conversation_id: str,
) -> bool:
    messages_file = _messages_path(base_dir, agent_id, user_id, conversation_id)
    existed = messages_file.exists()
    messages_file.unlink(missing_ok=True)

    list_file = _list_path(base_dir, agent_id, user_id)
    convs = _load_json(list_file, [])
    filtered = [c for c in convs if c.get("id") != conversation_id]
    if len(filtered) != len(convs):
        existed = True
        _write_json(list_file, filtered)
    return existed

