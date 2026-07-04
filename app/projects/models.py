"""Project & ProjectMember SQLAlchemy models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.config import get_config
from app.database import Base


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    slug: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    _disk_path: Mapped[str] = mapped_column("disk_path", String(512), nullable=False, default="")
    created_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    ticket_project_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True, default=None
    )
    feedback_enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")

    knowledge_api_enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    knowledge_api_model_name: Mapped[str] = mapped_column(String(128), default="", server_default="")
    knowledge_agent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True, default=None
    )

    git_repo_url: Mapped[str] = mapped_column(String(512), default="")
    git_branch: Mapped[str] = mapped_column(String(128), default="main")
    git_username: Mapped[str] = mapped_column(String(128), default="")
    git_auth_token: Mapped[str] = mapped_column(String(512), default="")
    git_author_name: Mapped[str] = mapped_column(String(128), default="")
    git_author_email: Mapped[str] = mapped_column(String(256), default="")
    git_sync_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    git_sync_auto_compile: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    git_sync_time: Mapped[str] = mapped_column(String(8), default="02:00")
    last_git_sync_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    last_git_sync_status: Mapped[str] = mapped_column(String(16), default="idle")
    last_git_sync_error: Mapped[str] = mapped_column(Text, default="")
    publish_repo_url: Mapped[str] = mapped_column(String(512), default="", server_default="")
    publish_branch: Mapped[str] = mapped_column(String(128), default="main", server_default="main")
    publish_username: Mapped[str] = mapped_column(String(128), default="", server_default="")
    publish_auth_token: Mapped[str] = mapped_column(String(512), default="", server_default="")
    publish_author_name: Mapped[str] = mapped_column(String(128), default="", server_default="")
    publish_author_email: Mapped[str] = mapped_column(String(256), default="", server_default="")
    publish_enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    last_publish_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    last_publish_status: Mapped[str] = mapped_column(String(16), default="idle", server_default="idle")
    last_publish_error: Mapped[str] = mapped_column(Text, default="", server_default="")
    ingest_paused: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    project_type: Mapped[str] = mapped_column(
        String(32), default="knowledge_base", server_default="knowledge_base"
    )
    case_index_auto_rebuild: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0"
    )

    @property
    def disk_path(self) -> str:
        return str(Path(get_config().server.projects_dir) / self.id)


class ProjectMember(Base):
    __tablename__ = "project_members"
    __table_args__ = (UniqueConstraint("project_id", "user_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="editor")


class ProjectSourceRepository(Base):
    __tablename__ = "project_source_repositories"
    __table_args__ = (UniqueConstraint("project_id", "key"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
    key: Mapped[str] = mapped_column(String(128), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    repo_url: Mapped[str] = mapped_column(String(512), default="", server_default="")
    branch: Mapped[str] = mapped_column(String(128), default="main", server_default="main")
    username: Mapped[str] = mapped_column(String(128), default="", server_default="")
    auth_token: Mapped[str] = mapped_column(String(512), default="", server_default="")
    author_name: Mapped[str] = mapped_column(String(128), default="", server_default="")
    author_email: Mapped[str] = mapped_column(String(256), default="", server_default="")
    sync_enabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    auto_compile: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    sync_time: Mapped[str] = mapped_column(String(8), default="02:00", server_default="02:00")
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, default=None)
    last_sync_status: Mapped[str] = mapped_column(String(16), default="idle", server_default="idle")
    last_sync_error: Mapped[str] = mapped_column(Text, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
