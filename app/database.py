"""Async SQLAlchemy database setup."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(get_settings().database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def init_db() -> None:
    from app.auth.models import User, UserApiToken  # noqa: F401
    from app.ingest.models import IngestJob  # noqa: F401
    from app.projects.models import Project, ProjectMember  # noqa: F401
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
    logger = logging.getLogger(__name__)

    migrations = [
        ("ingest_jobs", "step", "INTEGER DEFAULT 0"),
        ("projects", "ticket_project_id", "TEXT DEFAULT NULL REFERENCES projects(id) ON DELETE SET NULL"),
        ("projects", "feedback_enabled", "BOOLEAN DEFAULT 1"),
        ("agents", "max_tool_calls", "INTEGER DEFAULT 20"),
        ("agents", "debug_result_limit", "INTEGER DEFAULT 2000"),
        ("agents", "tool_labels", "TEXT DEFAULT '{}'"),
        ("projects", "git_repo_url", "TEXT DEFAULT ''"),
        ("projects", "git_branch", "TEXT DEFAULT 'main'"),
        ("projects", "git_username", "TEXT DEFAULT ''"),
        ("projects", "git_auth_token", "TEXT DEFAULT ''"),
        ("projects", "git_author_name", "TEXT DEFAULT ''"),
        ("projects", "git_author_email", "TEXT DEFAULT ''"),
        ("projects", "git_sync_enabled", "BOOLEAN DEFAULT 0"),
        ("projects", "git_sync_time", "TEXT DEFAULT '02:00'"),
        ("projects", "last_git_sync_at", "DATETIME DEFAULT NULL"),
        ("projects", "last_git_sync_status", "TEXT DEFAULT 'idle'"),
        ("projects", "last_git_sync_error", "TEXT DEFAULT ''"),
        ("projects", "ingest_paused", "BOOLEAN DEFAULT 0"),
        ("projects", "knowledge_api_enabled", "BOOLEAN DEFAULT 0"),
        ("projects", "knowledge_api_model_name", "TEXT DEFAULT ''"),
        ("projects", "knowledge_agent_id", "TEXT DEFAULT NULL REFERENCES agents(id) ON DELETE SET NULL"),
    ]

    for table, column, col_type in migrations:
        try:
            await conn.execute(
                __import__("sqlalchemy").text(
                    f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
                )
            )
            logger.info("Migrated: added %s.%s", table, column)
        except Exception:
            pass  # column already exists
