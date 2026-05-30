"""Unit tests for FeedbackTask model and status transitions."""

import pytest

from app.feedback.models import FeedbackTask, VALID_STATUSES, MAX_REVISIONS
from app.feedback.service import _TRANSITIONS, transition_status


class TestValidStatuses:
    def test_expected_count(self):
        assert len(VALID_STATUSES) == 8

    def test_required_statuses(self):
        expected = {
            "pending_evaluation", "evaluation_done", "pending_review",
            "pending_recompile", "approved", "rejected", "applied",
            "compile_failed",
        }
        assert VALID_STATUSES == expected


class TestTransitions:
    def test_all_sources_are_valid(self):
        for src in _TRANSITIONS:
            assert src in VALID_STATUSES, f"Source '{src}' not in VALID_STATUSES"

    def test_all_targets_are_valid(self):
        for src, targets in _TRANSITIONS.items():
            for t in targets:
                assert t in VALID_STATUSES, f"Target '{t}' from '{src}' not in VALID_STATUSES"

    def test_pending_evaluation_to_evaluation_done(self):
        task = _make_task("pending_evaluation")
        transition_status(task, "evaluation_done")
        assert task.status == "evaluation_done"

    def test_pending_evaluation_to_rejected(self):
        task = _make_task("pending_evaluation")
        transition_status(task, "rejected")
        assert task.status == "rejected"

    def test_pending_evaluation_to_compile_failed(self):
        task = _make_task("pending_evaluation")
        transition_status(task, "compile_failed")
        assert task.status == "compile_failed"

    def test_evaluation_done_to_pending_review(self):
        task = _make_task("evaluation_done")
        transition_status(task, "pending_review")
        assert task.status == "pending_review"

    def test_evaluation_done_to_pending_recompile(self):
        task = _make_task("evaluation_done")
        transition_status(task, "pending_recompile")
        assert task.status == "pending_recompile"

    def test_pending_review_to_approved(self):
        task = _make_task("pending_review")
        transition_status(task, "approved")
        assert task.status == "approved"

    def test_pending_review_to_rejected(self):
        task = _make_task("pending_review")
        transition_status(task, "rejected")
        assert task.status == "rejected"

    def test_pending_review_to_pending_recompile(self):
        task = _make_task("pending_review")
        transition_status(task, "pending_recompile")
        assert task.status == "pending_recompile"

    def test_approved_to_applied(self):
        task = _make_task("approved")
        transition_status(task, "applied")
        assert task.status == "applied"

    def test_invalid_transition_raises(self):
        task = _make_task("pending_evaluation")
        with pytest.raises(ValueError, match="Cannot transition"):
            transition_status(task, "approved")

    def test_applied_is_terminal(self):
        task = _make_task("applied")
        with pytest.raises(ValueError):
            transition_status(task, "pending_evaluation")


class TestFeedbackTaskColumns:
    def test_all_columns_present(self):
        cols = {c.name for c in FeedbackTask.__table__.columns}
        required = {
            "id", "project_id", "conversation_id", "agent_id",
            "user_message", "assistant_answer", "tool_traces_json",
            "wiki_reads_json", "raw_reads_json", "target_page_path",
            "page_exists", "evaluator_result_json", "evaluator_confidence",
            "repair_candidate_json", "status", "review_guidance",
            "reject_reason", "revision_count", "error",
            "created_at", "updated_at",
        }
        assert required.issubset(cols), f"Missing columns: {required - cols}"


class TestMaxRevisions:
    def test_default(self):
        assert MAX_REVISIONS == 5


class _FakeTask:
    """Lightweight stand-in for FeedbackTask in transition tests."""
    def __init__(self, status: str):
        self.status = status


def _make_task(status: str):
    """Create a minimal task-like object for testing transitions."""
    return _FakeTask(status)
