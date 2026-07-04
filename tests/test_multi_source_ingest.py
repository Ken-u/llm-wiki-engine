from __future__ import annotations

from types import SimpleNamespace

import httpx

from app.agents.models import Agent  # noqa: F401
from app.auth.models import User
from app.database import Base
from app.projects.models import Project, ProjectMember, ProjectSourceRepository
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


def test_remote_browser_preview_and_enqueue_are_scoped_by_source_repository(tmp_path, monkeypatch):
    async def run():
        from app.auth.deps import get_current_user
        from app.database import get_db
        from app.main import app

        projects_dir = tmp_path / "projects"
        monkeypatch.setattr(
            "app.projects.models.get_config",
            lambda: SimpleNamespace(server=SimpleNamespace(projects_dir=str(projects_dir))),
        )

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'multi-source.db'}")
        Session = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        project_dir = projects_dir / "project-1"
        frontend_root = project_dir / ".llm-wiki" / "source-repos" / "frontend"
        backend_root = project_dir / ".llm-wiki" / "source-repos" / "backend"
        frontend_root.mkdir(parents=True)
        backend_root.mkdir(parents=True)
        (frontend_root / "README.md").write_text("# Frontend\n", encoding="utf-8")
        (backend_root / "README.md").write_text("# Backend\n", encoding="utf-8")

        async with Session() as seed_db:
            owner = User(id=1, username="owner", password_hash="x", role="user")
            project = Project(id="project-1", name="Project", slug="project", description="", created_by=1)
            member = ProjectMember(project_id="project-1", user_id=1, role="owner")
            frontend = ProjectSourceRepository(
                id="frontend-id",
                project_id="project-1",
                key="frontend",
                name="Frontend",
                repo_url="https://git.example.com/org/frontend.git",
            )
            backend = ProjectSourceRepository(
                id="backend-id",
                project_id="project-1",
                key="backend",
                name="Backend",
                repo_url="https://git.example.com/org/backend.git",
            )
            seed_db.add_all([owner, project, member, frontend, backend])
            await seed_db.commit()

        enqueued: list[dict[str, object]] = []

        async def fake_enqueue_many(
            project_id: str,
            project_dir: str,
            source_paths: list[str],
            user_id: int,
        ) -> list[str]:
            job_ids: list[str] = []
            for source_path in source_paths:
                enqueued.append(
                    {
                        "project_id": project_id,
                        "project_dir": project_dir,
                        "source_path": source_path,
                        "user_id": user_id,
                    }
                )
                job_ids.append(f"job-{len(enqueued)}")
            return job_ids

        monkeypatch.setattr("app.ingest.enqueue_service.ingest_queue.enqueue_many", fake_enqueue_many)

        async def override_get_db():
            async with Session() as db:
                yield db

        async def override_get_current_user():
            return SimpleNamespace(id=1, username="owner", role="user")

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_current_user] = override_get_current_user
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                root_resp = await client.get(
                    "/api/projects/project-1/ingest/files",
                    params={"source_kind": "remote", "dir": ""},
                )
                assert root_resp.status_code == 200
                root_data = root_resp.json()
                assert root_data["items"] == []
                assert root_data["directories"] == [
                    {
                        "name": "Backend",
                        "path": "backend",
                        "source_repo_id": "backend-id",
                        "source_repo_key": "backend",
                        "source_repo_name": "Backend",
                        "kind": "source_repository",
                    },
                    {
                        "name": "Frontend",
                        "path": "frontend",
                        "source_repo_id": "frontend-id",
                        "source_repo_key": "frontend",
                        "source_repo_name": "Frontend",
                        "kind": "source_repository",
                    },
                ]

                frontend_resp = await client.get(
                    "/api/projects/project-1/ingest/files",
                    params={
                        "source_kind": "remote",
                        "source_repo_id": "frontend-id",
                        "dir": "",
                    },
                )
                assert frontend_resp.status_code == 200
                frontend_data = frontend_resp.json()
                assert frontend_data["directories"] == []
                assert [item["source_file"] for item in frontend_data["items"]] == ["README.md"]
                assert frontend_data["items"][0]["source_repo_id"] == "frontend-id"
                assert frontend_data["items"][0]["source_repo_key"] == "frontend"
                assert frontend_data["items"][0]["source_path"].endswith(
                    ".llm-wiki/source-repos/frontend/README.md"
                )

                preview_resp = await client.get(
                    "/api/projects/project-1/ingest/files/content",
                    params={
                        "source_kind": "remote",
                        "source_repo_id": "frontend-id",
                        "source_file": "README.md",
                    },
                )
                assert preview_resp.status_code == 200
                assert preview_resp.json()["content"] == "# Frontend\n"

                missing_repo_preview = await client.get(
                    "/api/projects/project-1/ingest/files/content",
                    params={"source_kind": "remote", "source_file": "README.md"},
                )
                assert missing_repo_preview.status_code == 400

                traversal_preview = await client.get(
                    "/api/projects/project-1/ingest/files/content",
                    params={
                        "source_kind": "remote",
                        "source_repo_id": "frontend-id",
                        "source_file": "../backend/README.md",
                    },
                )
                assert traversal_preview.status_code == 400

                enqueue_resp = await client.post(
                    "/api/projects/project-1/ingest/files/enqueue",
                    json={
                        "source_kind": "remote",
                        "source_repo_id": "frontend-id",
                        "source_files": ["README.md"],
                    },
                )
                assert enqueue_resp.status_code == 202
                assert enqueue_resp.json() == {"jobs": ["job-1"], "count": 1}
                assert enqueued[0]["source_path"] == str(
                    project_dir / "raw" / "sources" / "frontend" / "README.md"
                )
                assert (project_dir / "raw" / "sources" / "frontend" / "README.md").read_text(
                    encoding="utf-8"
                ) == "# Frontend\n"

                enqueue_backend_resp = await client.post(
                    "/api/projects/project-1/ingest/files/enqueue",
                    json={
                        "source_kind": "remote",
                        "source_repo_id": "backend-id",
                        "source_files": ["README.md"],
                    },
                )
                assert enqueue_backend_resp.status_code == 202
                assert enqueued[1]["source_path"] == str(
                    project_dir / "raw" / "sources" / "backend" / "README.md"
                )
                assert (project_dir / "raw" / "sources" / "backend" / "README.md").read_text(
                    encoding="utf-8"
                ) == "# Backend\n"

                source_map = (project_dir / ".llm-wiki" / "source-map.json").read_text(encoding="utf-8")
                assert '"frontend/README.md"' in source_map
                assert '"backend/README.md"' in source_map
                assert '"source_repo_id": "frontend-id"' in source_map
                assert '"source_repo_key": "frontend"' in source_map
                assert '"source_repo_id": "backend-id"' in source_map
                assert '"source_repo_key": "backend"' in source_map
        finally:
            app.dependency_overrides.pop(get_db, None)
            app.dependency_overrides.pop(get_current_user, None)
            await engine.dispose()

    import asyncio

    asyncio.run(run())
