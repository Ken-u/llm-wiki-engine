"""Tests for SQLite engine concurrency settings."""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.database import _auto_migrate, _engine_kwargs


def test_sqlite_engine_uses_busy_timeout_connect_arg():
    kwargs = _engine_kwargs("sqlite+aiosqlite:///./data/engine.db")

    assert kwargs["connect_args"]["timeout"] == 30


def test_non_sqlite_engine_has_no_sqlite_connect_args():
    assert _engine_kwargs("postgresql+asyncpg://example/db") == {}


def test_auto_migrate_moves_legacy_git_config_to_publish_config(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}")
        async with engine.begin() as conn:
            await conn.execute(text("""
                CREATE TABLE projects (
                    id TEXT PRIMARY KEY,
                    git_repo_url TEXT DEFAULT '',
                    git_branch TEXT DEFAULT 'main',
                    git_username TEXT DEFAULT '',
                    git_auth_token TEXT DEFAULT '',
                    git_author_name TEXT DEFAULT '',
                    git_author_email TEXT DEFAULT '',
                    git_sync_enabled BOOLEAN DEFAULT 0,
                    git_sync_auto_compile BOOLEAN DEFAULT 0,
                    git_sync_time TEXT DEFAULT '02:00',
                    last_git_sync_status TEXT DEFAULT 'idle',
                    last_git_sync_error TEXT DEFAULT ''
                )
            """))
            await conn.execute(text("""
                INSERT INTO projects (
                    id, git_repo_url, git_branch, git_username, git_auth_token,
                    git_author_name, git_author_email, git_sync_enabled,
                    git_sync_auto_compile, last_git_sync_status, last_git_sync_error
                )
                VALUES (
                    'p1', 'ssh://gerrit/project', 'master', 'bot', 'secret',
                    'Bot', 'bot@example.com', 1, 1, 'failed', 'old error'
                )
            """))
            await _auto_migrate(conn)
            row = (await conn.execute(text("""
                SELECT
                    git_repo_url, git_username, git_auth_token, git_sync_enabled,
                    git_sync_auto_compile, last_git_sync_status, last_git_sync_error,
                    publish_repo_url, publish_branch, publish_username, publish_auth_token,
                    publish_author_name, publish_author_email, publish_enabled
                FROM projects WHERE id = 'p1'
            """))).mappings().one()
            marker = (await conn.execute(text("""
                SELECT data FROM system_settings WHERE section = 'migration:legacy_git_to_publish'
            """))).scalar_one()
        await engine.dispose()
        return row, marker

    row, marker = asyncio.run(run())

    assert row["publish_repo_url"] == "ssh://gerrit/project"
    assert row["publish_branch"] == "master"
    assert row["publish_username"] == "bot"
    assert row["publish_auth_token"] == "secret"
    assert row["publish_author_name"] == "Bot"
    assert row["publish_author_email"] == "bot@example.com"
    assert row["publish_enabled"] == 1
    assert row["git_repo_url"] == ""
    assert row["git_username"] == ""
    assert row["git_auth_token"] == ""
    assert row["git_sync_enabled"] == 0
    assert row["git_sync_auto_compile"] == 0
    assert row["last_git_sync_status"] == "idle"
    assert row["last_git_sync_error"] == ""
    assert marker == "{}"
