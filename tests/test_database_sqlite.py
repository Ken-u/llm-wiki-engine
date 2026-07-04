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
                    'p1', 'https://git.example.com/org/docs.git', 'master', 'bot', 'secret',
                    'Bot', 'bot@example.com', 1, 1, 'failed', 'old error'
                )
            """))
            await conn.execute(text("""
                INSERT INTO projects (
                    id, git_repo_url, git_branch, git_username, git_auth_token,
                    git_author_name, git_author_email, git_sync_enabled,
                    git_sync_auto_compile, last_git_sync_status, last_git_sync_error
                )
                VALUES (
                    'p2', 'git@git.example.com:org/backend-docs.git', 'main', '', '',
                    '', '', 0, 0, 'idle', ''
                )
            """))
            await _auto_migrate(conn)
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
            source_marker = (await conn.execute(text("""
                SELECT data FROM system_settings WHERE section = 'migration:legacy_git_to_source_repositories'
            """))).scalar_one()
            source_repo = (await conn.execute(text("""
                SELECT
                    project_id, key, name, repo_url, branch, username, auth_token,
                    author_name, author_email, sync_enabled, auto_compile,
                    sync_time, last_sync_status, last_sync_error
                FROM project_source_repositories
                WHERE project_id = 'p1' AND key = 'default'
            """))).mappings().one()
            source_repo_count = (await conn.execute(text("""
                SELECT COUNT(*)
                FROM project_source_repositories
                WHERE project_id = 'p1' AND key = 'default'
            """))).scalar_one()
            ssh_source_repo_name = (await conn.execute(text("""
                SELECT name
                FROM project_source_repositories
                WHERE project_id = 'p2' AND key = 'default'
            """))).scalar_one()
        await engine.dispose()
        return row, marker, source_marker, source_repo, source_repo_count, ssh_source_repo_name

    row, marker, source_marker, source_repo, source_repo_count, ssh_source_repo_name = asyncio.run(run())

    assert row["publish_repo_url"] == "https://git.example.com/org/docs.git"
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
    assert source_marker == "{}"
    assert source_repo["project_id"] == "p1"
    assert source_repo["key"] == "default"
    assert source_repo["name"] == "docs"
    assert source_repo["repo_url"] == "https://git.example.com/org/docs.git"
    assert source_repo["branch"] == "master"
    assert source_repo["username"] == "bot"
    assert source_repo["auth_token"] == "secret"
    assert source_repo["author_name"] == "Bot"
    assert source_repo["author_email"] == "bot@example.com"
    assert source_repo["sync_enabled"] == 1
    assert source_repo["auto_compile"] == 1
    assert source_repo["sync_time"] == "02:00"
    assert source_repo["last_sync_status"] == "failed"
    assert source_repo["last_sync_error"] == "old error"
    assert source_repo_count == 1
    assert ssh_source_repo_name == "backend-docs"
