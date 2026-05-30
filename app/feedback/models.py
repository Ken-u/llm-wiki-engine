"""FeedbackTask SQLAlchemy model for the feedback recompile queue."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

VALID_STATUSES = {
    "pending_evaluation",
    "evaluation_done",
    "pending_review",
    "pending_recompile",
    "approved",
    "rejected",
    "applied",
    "compile_failed",
}

MAX_REVISIONS = 5


class FeedbackTask(Base):
    __tablename__ = "feedback_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    conversation_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("agents.id", ondelete="SET NULL"), nullable=True
    )

    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_answer: Mapped[str] = mapped_column(Text, nullable=False)
    tool_traces_json: Mapped[str] = mapped_column(Text, default="[]")
    wiki_reads_json: Mapped[str] = mapped_column(Text, default="[]")
    raw_reads_json: Mapped[str] = mapped_column(Text, default="[]")

    target_page_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page_exists: Mapped[bool] = mapped_column(Boolean, default=True)
    evaluator_result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluator_confidence: Mapped[str | None] = mapped_column(String(16), nullable=True)

    repair_candidate_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending_evaluation", index=True)
    review_guidance: Mapped[str | None] = mapped_column(Text, nullable=True)
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    revision_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
