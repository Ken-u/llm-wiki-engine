"""Async SQLAlchemy database setup."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _engine_kwargs(database_url: str) -> dict:
    if not database_url.startswith("sqlite+aiosqlite"):
        return {}
    return {
        "connect_args": {
            "timeout": 30,
        },
    }


engine = create_async_engine(
    get_settings().database_url,
    echo=False,
    **_engine_kwargs(get_settings().database_url),
)


@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
    if not get_settings().database_url.startswith("sqlite+aiosqlite"):
        return
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def init_db() -> None:
    from app.auth.models import User, UserApiToken  # noqa: F401
    from app.ingest.models import IngestJob  # noqa: F401
    from app.projects.models import Project, ProjectMember, ProjectSourceRepository  # noqa: F401
    from app.agents.models import Agent, AgentProject  # noqa: F401
    from app.feedback.models import FeedbackTask  # noqa: F401

    Path = __import__("pathlib").Path
    db_path = get_settings().database_url.replace("sqlite+aiosqlite:///", "")
    if db_path.startswith("./"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _auto_migrate(conn)
        await _ensure_system_settings_table(conn)

    await _seed_admin()

    from app.config import load_db_overrides
    load_db_overrides()


async def _seed_admin() -> None:
    """Create or update the admin account. Password read from ADMIN_PASSWORD env var."""
    import os
    import logging
    from sqlalchemy import select as sa_select
    logger = logging.getLogger(__name__)

    password = os.environ.get("ADMIN_PASSWORD", "admin")

    async with async_session() as db:
        from app.auth.models import User
        from app.auth.deps import hash_password, verify_password
        admin = (await db.execute(sa_select(User).where(User.username == "admin"))).scalar_one_or_none()
        if admin is None:
            db.add(User(username="admin", password_hash=hash_password(password), role="admin"))
            await db.commit()
            logger.info("Created admin account (password from ADMIN_PASSWORD)")
        else:
            changed = False
            if admin.role != "admin":
                admin.role = "admin"
                changed = True
            if not verify_password(password, admin.password_hash):
                admin.password_hash = hash_password(password)
                changed = True
                logger.info("Admin password updated from ADMIN_PASSWORD")
            if changed:
                await db.commit()


async def _ensure_system_settings_table(conn) -> None:
    """Create system_settings table for admin config persistence."""
    import sqlalchemy
    await conn.execute(sqlalchemy.text("""
        CREATE TABLE IF NOT EXISTS system_settings (
            section TEXT PRIMARY KEY,
            data    TEXT NOT NULL DEFAULT '{}'
        )
    """))


async def _auto_migrate(conn) -> None:
    """Add missing columns to existing tables (lightweight SQLite migration)."""
    import logging
    import sqlalchemy
    logger = logging.getLogger(__name__)

    migrations = [
        ("ingest_jobs", "step", "INTEGER DEFAULT 0"),
        ("projects", "ticket_project_id", "TEXT DEFAULT NULL REFERENCES projects(id) ON DELETE SET NULL"),
        ("projects", "feedback_enabled", "BOOLEAN DEFAULT 1"),
        ("agents", "max_tool_calls", "INTEGER DEFAULT 20"),
        ("agents", "debug_result_limit", "INTEGER DEFAULT 2000"),
        ("agents", "tool_labels", "TEXT DEFAULT '{}'"),
        ("agents", "system_prompt_override", "TEXT DEFAULT ''"),
        ("projects", "git_repo_url", "TEXT DEFAULT ''"),
        ("projects", "git_branch", "TEXT DEFAULT 'main'"),
        ("projects", "git_username", "TEXT DEFAULT ''"),
        ("projects", "git_auth_token", "TEXT DEFAULT ''"),
        ("projects", "git_author_name", "TEXT DEFAULT ''"),
        ("projects", "git_author_email", "TEXT DEFAULT ''"),
        ("projects", "git_sync_enabled", "BOOLEAN DEFAULT 0"),
        ("projects", "git_sync_auto_compile", "BOOLEAN DEFAULT 0"),
        ("projects", "git_sync_time", "TEXT DEFAULT '02:00'"),
        ("projects", "last_git_sync_at", "DATETIME DEFAULT NULL"),
        ("projects", "last_git_sync_status", "TEXT DEFAULT 'idle'"),
        ("projects", "last_git_sync_error", "TEXT DEFAULT ''"),
        ("projects", "publish_repo_url", "TEXT DEFAULT ''"),
        ("projects", "publish_branch", "TEXT DEFAULT 'main'"),
        ("projects", "publish_username", "TEXT DEFAULT ''"),
        ("projects", "publish_auth_token", "TEXT DEFAULT ''"),
        ("projects", "publish_author_name", "TEXT DEFAULT ''"),
        ("projects", "publish_author_email", "TEXT DEFAULT ''"),
        ("projects", "publish_enabled", "BOOLEAN DEFAULT 0"),
        ("projects", "last_publish_at", "DATETIME DEFAULT NULL"),
        ("projects", "last_publish_status", "TEXT DEFAULT 'idle'"),
        ("projects", "last_publish_error", "TEXT DEFAULT ''"),
        ("projects", "ingest_paused", "BOOLEAN DEFAULT 0"),
        ("projects", "knowledge_api_enabled", "BOOLEAN DEFAULT 0"),
        ("projects", "knowledge_api_model_name", "TEXT DEFAULT ''"),
        ("projects", "knowledge_agent_id", "TEXT DEFAULT NULL REFERENCES agents(id) ON DELETE SET NULL"),
        ("projects", "project_type", "TEXT DEFAULT 'knowledge_base'"),
        ("projects", "case_index_auto_rebuild", "BOOLEAN DEFAULT 0"),
    ]

    for table, column, col_type in migrations:
        try:
            await conn.execute(
                sqlalchemy.text(
                    f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
                )
            )
            logger.info("Migrated: added %s.%s", table, column)
        except Exception:
            pass  # column already exists

    await conn.execute(sqlalchemy.text("""
        CREATE TABLE IF NOT EXISTS project_source_repositories (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
            key TEXT NOT NULL,
            name TEXT NOT NULL,
            repo_url TEXT DEFAULT '',
            branch TEXT DEFAULT 'main',
            username TEXT DEFAULT '',
            auth_token TEXT DEFAULT '',
            author_name TEXT DEFAULT '',
            author_email TEXT DEFAULT '',
            sync_enabled BOOLEAN DEFAULT 0,
            auto_compile BOOLEAN DEFAULT 0,
            sync_time TEXT DEFAULT '02:00',
            last_sync_at DATETIME DEFAULT NULL,
            last_sync_status TEXT DEFAULT 'idle',
            last_sync_error TEXT DEFAULT '',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(project_id, key)
        )
    """))

    await _ensure_system_settings_table(conn)
    await _migrate_legacy_git_config_to_source_repositories(conn)
    await _migrate_legacy_git_config_to_publish(conn)


async def _migrate_legacy_git_config_to_source_repositories(conn) -> None:
    """Create default source repositories from legacy project.git_* config."""
    import logging
    import sqlalchemy
    logger = logging.getLogger(__name__)
    marker = "migration:legacy_git_to_source_repositories"
    existing = (
        await conn.execute(
            sqlalchemy.text("SELECT data FROM system_settings WHERE section = :section"),
            {"section": marker},
        )
    ).scalar_one_or_none()
    if existing is not None:
        return

    result = await conn.execute(
        sqlalchemy.text("""
            WITH legacy_projects AS (
                SELECT
                    p.*,
                    CASE
                        WHEN instr(trim(p.git_repo_url), '/') > 0 THEN trim(p.git_repo_url)
                        WHEN instr(trim(p.git_repo_url), ':') > 0 THEN substr(trim(p.git_repo_url), instr(trim(p.git_repo_url), ':') + 1)
                        ELSE trim(p.git_repo_url)
                    END AS split_source
                FROM projects p
                WHERE COALESCE(p.git_repo_url, '') != ''
                  AND NOT EXISTS (
                      SELECT 1
                      FROM project_source_repositories existing
                      WHERE existing.project_id = p.id
                        AND existing.key = 'default'
                  )
            ),
            name_parts(project_id, rest, segment) AS (
                SELECT
                    id,
                    split_source || '/',
                    ''
                FROM legacy_projects
                UNION ALL
                SELECT
                    project_id,
                    substr(rest, instr(rest, '/') + 1),
                    substr(rest, 1, instr(rest, '/') - 1)
                FROM name_parts
                WHERE rest != ''
            ),
            inferred_names AS (
                SELECT
                    project_id,
                    CASE
                        WHEN segment LIKE '%.git' AND substr(segment, 1, length(segment) - 4) != ''
                            THEN substr(segment, 1, length(segment) - 4)
                        WHEN segment != ''
                            THEN segment
                        ELSE '默认源仓库'
                    END AS repo_name
                FROM name_parts
                WHERE rest = ''
            )
            INSERT INTO project_source_repositories (
                id, project_id, key, name, repo_url, branch, username, auth_token,
                author_name, author_email, sync_enabled, auto_compile, sync_time,
                last_sync_at, last_sync_status, last_sync_error
            )
            SELECT
                lower(hex(randomblob(4))) || '-' ||
                    lower(hex(randomblob(2))) || '-' ||
                    lower(hex(randomblob(2))) || '-' ||
                    lower(hex(randomblob(2))) || '-' ||
                    lower(hex(randomblob(6))),
                p.id,
                'default',
                inferred_names.repo_name,
                p.git_repo_url,
                COALESCE(NULLIF(p.git_branch, ''), 'main'),
                COALESCE(p.git_username, ''),
                COALESCE(p.git_auth_token, ''),
                COALESCE(p.git_author_name, ''),
                COALESCE(p.git_author_email, ''),
                COALESCE(p.git_sync_enabled, 0),
                COALESCE(p.git_sync_auto_compile, 0),
                COALESCE(NULLIF(p.git_sync_time, ''), '02:00'),
                p.last_git_sync_at,
                COALESCE(NULLIF(p.last_git_sync_status, ''), 'idle'),
                COALESCE(p.last_git_sync_error, '')
            FROM legacy_projects p
            JOIN inferred_names ON inferred_names.project_id = p.id
        """)
    )
    await conn.execute(
        sqlalchemy.text("INSERT INTO system_settings(section, data) VALUES (:section, :data)"),
        {"section": marker, "data": "{}"},
    )
    logger.info("Migrated legacy Git config to default source repositories for %d projects", result.rowcount or 0)


async def _migrate_legacy_git_config_to_publish(conn) -> None:
    """Treat pre-source-repo Git config as publish-repo config.

    Before source repositories existed, project.git_* described the LLM Wiki
    project repository itself. After the split, those values belong to
    publish_* and source Git config should start blank.
    """
    import logging
    import sqlalchemy
    logger = logging.getLogger(__name__)
    marker = "migration:legacy_git_to_publish"
    existing = (
        await conn.execute(
            sqlalchemy.text("SELECT data FROM system_settings WHERE section = :section"),
            {"section": marker},
        )
    ).scalar_one_or_none()
    if existing is not None:
        return

    result = await conn.execute(
        sqlalchemy.text("""
            UPDATE projects
            SET
                publish_repo_url = git_repo_url,
                publish_branch = COALESCE(NULLIF(git_branch, ''), 'main'),
                publish_username = git_username,
                publish_auth_token = git_auth_token,
                publish_author_name = git_author_name,
                publish_author_email = git_author_email,
                publish_enabled = git_sync_enabled,
                git_repo_url = '',
                git_username = '',
                git_auth_token = '',
                git_sync_enabled = 0,
                git_sync_auto_compile = 0,
                last_git_sync_status = 'idle',
                last_git_sync_error = ''
            WHERE COALESCE(git_repo_url, '') != ''
              AND COALESCE(publish_repo_url, '') = ''
        """)
    )
    await conn.execute(
        sqlalchemy.text("INSERT INTO system_settings(section, data) VALUES (:section, :data)"),
        {"section": marker, "data": "{}"},
    )
    logger.info("Migrated legacy Git config to publish config for %d projects", result.rowcount or 0)
