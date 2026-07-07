"""Tests for publishing after feedback-applied wiki changes."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.feedback.router import _apply_wiki_changes


class _ScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _FakeDb:
    def __init__(self, project):
        self.project = project

    async def execute(self, _stmt):
        return _ScalarResult(self.project)


def test_apply_wiki_changes_auto_publishes_when_content_changes(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    project = SimpleNamespace(disk_path=str(tmp_path))
    publisher = AsyncMock()

    async def run():
        with patch("app.feedback.router.maybe_auto_publish_project_to_git", publisher, create=True):
            await _apply_wiki_changes(
                _FakeDb(project),
                "project-1",
                [{"path": "page.md", "action": "modify", "new_content": "# Updated\n"}],
                "task-12345678",
            )

    asyncio.run(run())

    publisher.assert_awaited_once_with(
        "project-1",
        triggered_by=0,
        reason="feedback:task-12345678",
    )


def test_apply_wiki_changes_skips_auto_publish_when_content_is_unchanged(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "page.md").write_text("# Same\n", encoding="utf-8")
    project = SimpleNamespace(disk_path=str(tmp_path))
    publisher = AsyncMock()

    async def run():
        with patch("app.feedback.router.maybe_auto_publish_project_to_git", publisher, create=True):
            await _apply_wiki_changes(
                _FakeDb(project),
                "project-1",
                [{"path": "page.md", "action": "modify", "new_content": "# Same\n"}],
                "task-12345678",
            )

    asyncio.run(run())

    publisher.assert_not_awaited()
