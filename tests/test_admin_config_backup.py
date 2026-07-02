import asyncio
from types import SimpleNamespace

from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.admin.router import _export_config_bundle, _restore_config_bundle
from app.agents.models import Agent, AgentProject
from app.auth.models import User
from app.database import Base
from app.projects.models import Project, ProjectMember


def test_export_config_bundle_contains_only_config_state(tmp_path):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'export.db'}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS system_settings (
                    section TEXT PRIMARY KEY,
                    data    TEXT NOT NULL DEFAULT '{}'
                )
            """))
        Session = async_sessionmaker(engine, expire_on_commit=False)

        async with Session() as db:
            db.add(User(id=1, username="admin", password_hash="hash-1", role="admin"))
            db.add(Project(
                id="proj-1", name="Proj", slug="proj", description="desc",
                _disk_path="legacy/path", created_by=1, feedback_enabled=True,
            ))
            db.add(ProjectMember(project_id="proj-1", user_id=1, role="owner"))
            db.add(Agent(
                id="agent-1", name="Agent", description="desc", system_prompt="hi",
                system_prompt_override="full prompt",
                is_public=True, require_api_key=True, api_key_hash="keyhash",
                max_tool_calls=9, debug_result_limit=99, tool_labels='{"a":"b"}', created_by=1,
            ))
            db.add(AgentProject(agent_id="agent-1", project_id="proj-1"))
            await db.execute(
                text("INSERT INTO system_settings(section, data) VALUES (:section, :data)"),
                {"section": "llm", "data": '{"model":"gpt-test"}'},
            )
            await db.commit()

            bundle = await _export_config_bundle(db)

        await engine.dispose()
        return bundle

    bundle = asyncio.run(run())
    assert bundle["version"] == 1
    assert bundle["users"][0]["username"] == "admin"
    assert bundle["projects"][0]["slug"] == "proj"
    assert "disk_path" not in bundle["projects"][0]
    assert bundle["project_members"][0]["role"] == "owner"
    assert bundle["agents"][0]["api_key_hash"] == "keyhash"
    assert bundle["agents"][0]["system_prompt_override"] == "full prompt"
    assert bundle["agent_projects"][0]["project_id"] == "proj-1"
    assert bundle["system_settings"][0]["section"] == "llm"
    assert "ingest_jobs" not in bundle


def test_restore_config_bundle_replaces_existing_config_state(tmp_path, monkeypatch):
    async def run():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'import.db'}")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS system_settings (
                    section TEXT PRIMARY KEY,
                    data    TEXT NOT NULL DEFAULT '{}'
                )
            """))
        Session = async_sessionmaker(engine, expire_on_commit=False)

        monkeypatch.setattr(
            "app.projects.models.get_config",
            lambda: SimpleNamespace(server=SimpleNamespace(projects_dir="/tmp/dev-projects")),
        )
        monkeypatch.setattr("app.admin.router._reload_config", lambda: None)
        monkeypatch.setattr("app.admin.router.load_db_overrides", lambda: None)

        bundle = {
            "version": 1,
            "users": [
                {"id": 10, "username": "admin", "password_hash": "hash-admin", "role": "admin", "created_at": "2026-06-04T00:00:00"},
                {"id": 11, "username": "alice", "password_hash": "hash-alice", "role": "user", "created_at": "2026-06-04T00:00:01"},
            ],
            "system_settings": [
                {"section": "llm", "data": {"model": "gpt-4o-mini"}},
            ],
            "projects": [
                {
                    "id": "proj-1",
                    "name": "Proj",
                    "slug": "proj",
                    "description": "desc",
                    "created_by": 10,
                    "created_at": "2026-06-04T00:00:02",
                    "ticket_project_id": None,
                    "feedback_enabled": True,
                    "git_repo_url": "https://git.example/repo.git",
                    "git_branch": "main",
                    "git_username": "bot",
                    "git_auth_token": "secret",
                    "git_author_name": "Bot",
                    "git_author_email": "bot@example.com",
                    "git_sync_enabled": True,
                    "git_sync_time": "03:00",
                }
            ],
            "project_members": [
                {"project_id": "proj-1", "user_id": 10, "role": "owner"},
                {"project_id": "proj-1", "user_id": 11, "role": "editor"},
            ],
            "agents": [
                {
                    "id": "agent-1",
                    "name": "Agent",
                    "description": "desc",
                    "system_prompt": "hello",
                    "system_prompt_override": "override prompt",
                    "is_public": False,
                    "require_api_key": True,
                    "api_key_hash": "api-hash",
                    "max_tool_calls": 20,
                    "debug_result_limit": 2000,
                    "tool_labels": '{"search":"Search"}',
                    "created_by": 10,
                    "created_at": "2026-06-04T00:00:03",
                }
            ],
            "agent_projects": [
                {"agent_id": "agent-1", "project_id": "proj-1"},
            ],
        }

        async with Session() as db:
            db.add(User(id=1, username="stale", password_hash="old", role="user"))
            db.add(Project(id="old-proj", name="Old", slug="old", description="", _disk_path="old", created_by=1))
            await db.commit()

            summary = await _restore_config_bundle(db, bundle)
            assert summary["users"] == 2
            assert summary["projects"] == 1
            assert summary["agents"] == 1

            users = list((await db.execute(select(User).order_by(User.id))).scalars().all())
            projects = list((await db.execute(select(Project))).scalars().all())
            members = list((await db.execute(select(ProjectMember).order_by(ProjectMember.user_id))).scalars().all())
            agents = list((await db.execute(select(Agent))).scalars().all())
            links = list((await db.execute(select(AgentProject))).scalars().all())
            rows = (await db.execute(text("SELECT section, data FROM system_settings"))).fetchall()

        await engine.dispose()
        return users, projects, members, agents, links, rows

    users, projects, members, agents, links, rows = asyncio.run(run())
    assert [u.username for u in users] == ["admin", "alice"]
    assert projects[0].slug == "proj"
    assert projects[0].disk_path == "/tmp/dev-projects/proj-1"
    assert [m.role for m in members] == ["owner", "editor"]
    assert agents[0].name == "Agent"
    assert agents[0].system_prompt_override == "override prompt"
    assert links[0].project_id == "proj-1"
    assert rows[0][0] == "llm"
