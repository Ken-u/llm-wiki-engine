"""Tests for SQLite engine concurrency settings."""

from app.database import _engine_kwargs


def test_sqlite_engine_uses_busy_timeout_connect_arg():
    kwargs = _engine_kwargs("sqlite+aiosqlite:///./data/engine.db")

    assert kwargs["connect_args"]["timeout"] == 30


def test_non_sqlite_engine_has_no_sqlite_connect_args():
    assert _engine_kwargs("postgresql+asyncpg://example/db") == {}
