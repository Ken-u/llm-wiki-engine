"""Unit tests for per-source ingest file status helpers."""

from datetime import datetime, timezone
from types import SimpleNamespace

from app.ingest.files import (
    IngestFileItem,
    filter_and_sort_items,
    paginate_items,
    resolve_file_statuses,
)


def test_resolve_file_statuses_assigns_each_file_to_one_highest_priority_status():
    now = datetime.now(timezone.utc)
    jobs = [
        SimpleNamespace(
            source_path="/project/raw/sources/a.md",
            status="done",
            files_written=["wiki/a.md"],
            error=None,
            progress="Complete",
            step=3,
            created_at=now,
            completed_at=now,
        ),
        SimpleNamespace(
            source_path="/project/raw/sources/a.md",
            status="pending",
            files_written=None,
            error=None,
            progress="Queued",
            step=0,
            created_at=now,
            completed_at=None,
        ),
        SimpleNamespace(
            source_path="/project/raw/sources/b.md",
            status="done",
            files_written=["wiki/b.md"],
            error=None,
            progress="Complete",
            step=3,
            created_at=now,
            completed_at=now,
        ),
        SimpleNamespace(
            source_path="/project/raw/sources/d.md",
            status="failed",
            files_written=None,
            error="upstream error",
            progress="Failed",
            step=1,
            created_at=now,
            completed_at=now,
        ),
    ]

    items = resolve_file_statuses(
        source_paths=[
            "/project/raw/sources/a.md",
            "/project/raw/sources/b.md",
            "/project/raw/sources/c.md",
            "/project/raw/sources/d.md",
        ],
        jobs=jobs,
        changed_identities={"b.md"},
        cached_identities={"a.md", "b.md"},
    )

    by_name = {item.source_file: item for item in items}
    assert by_name["a.md"].status == "queued"
    assert by_name["b.md"].status == "updated"
    assert by_name["c.md"].status == "not_queued"
    assert by_name["d.md"].status == "failed"
    assert [item.source_file for item in items].count("a.md") == 1


def test_paginate_items_defaults_to_first_ten_and_reports_total():
    items = [
        IngestFileItem(source_file=f"{idx}.md", source_path=f"/tmp/{idx}.md", status="not_queued")
        for idx in range(12)
    ]

    page = paginate_items(items, page=1, page_size=10)

    assert page.total == 12
    assert page.page == 1
    assert page.page_size == 10
    assert [item.source_file for item in page.items] == [f"{idx}.md" for idx in range(10)]


def test_resolve_file_statuses_uses_relative_paths_for_nested_sources():
    items = resolve_file_statuses(
        source_paths=[
            "/project/raw/sources/a/guide.md",
            "/project/raw/sources/b/guide.md",
        ],
        jobs=[],
        changed_identities={"b/guide.md"},
        cached_identities={"a/guide.md", "b/guide.md"},
        source_root="/project/raw/sources",
    )

    by_name = {item.source_file: item for item in items}
    assert by_name["a/guide.md"].status == "compiled"
    assert by_name["b/guide.md"].status == "updated"


def test_resolve_file_statuses_does_not_put_paused_backlog_in_processing():
    now = datetime.now(timezone.utc)
    jobs = [
        SimpleNamespace(
            source_path="/project/raw/sources/a.md",
            status="paused",
            files_written=None,
            error=None,
            progress="Paused",
            step=0,
            created_at=now,
            completed_at=None,
        ),
    ]

    items = resolve_file_statuses(
        source_paths=["/project/raw/sources/a.md"],
        jobs=jobs,
        changed_identities=set(),
        cached_identities=set(),
    )

    assert items[0].status == "queued"


def test_filter_and_sort_items_searches_before_name_sorting():
    items = [
        IngestFileItem(source_file="beta.md", source_path="/tmp/beta.md", status="not_queued"),
        IngestFileItem(source_file="alpha-guide.md", source_path="/tmp/alpha-guide.md", status="not_queued"),
        IngestFileItem(source_file="z-alpha.md", source_path="/tmp/z-alpha.md", status="not_queued"),
    ]

    filtered = filter_and_sort_items(items, search="alpha", sort_dir="desc")

    assert [item.source_file for item in filtered] == ["z-alpha.md", "alpha-guide.md"]
