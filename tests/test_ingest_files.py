"""Unit tests for per-source ingest file status helpers."""

from datetime import datetime, timezone
from types import SimpleNamespace

from app.ingest.files import (
    IngestFileItem,
    IngestSelection,
    apply_selection,
    browser_source_root,
    filter_and_sort_items,
    list_project_source_files,
    list_source_files_at_root,
    load_source_map,
    local_source_root,
    paginate_items,
    provider_source_root,
    resolve_file_statuses,
    source_identity,
    stage_source_for_ingest,
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


def test_resolve_file_statuses_uses_new_success_over_old_failure():
    old = datetime(2026, 6, 10, 11, 36, tzinfo=timezone.utc)
    new = datetime(2026, 6, 10, 14, 13, tzinfo=timezone.utc)
    jobs = [
        SimpleNamespace(
            source_path="/project/raw/sources/guide.md",
            status="failed",
            files_written=None,
            error="old upstream error",
            progress="Failed",
            step=0,
            created_at=old,
            completed_at=old,
        ),
        SimpleNamespace(
            source_path="/project/raw/sources/guide.md",
            status="done",
            files_written=["wiki/sources/guide.md"],
            error=None,
            progress="Complete",
            step=3,
            created_at=new,
            completed_at=new,
        ),
    ]

    items = resolve_file_statuses(
        source_paths=["/project/raw/sources/guide.md"],
        jobs=jobs,
        changed_identities=set(),
        cached_identities={"guide.md"},
    )

    assert items[0].status == "compiled"
    assert items[0].job_status == "done"
    assert items[0].error is None


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


def test_apply_selection_filters_by_status_globs_excludes_and_query():
    items = [
        IngestFileItem(source_file="docs/a.md", source_path="/tmp/docs/a.md", status="updated"),
        IngestFileItem(source_file="docs/draft/b.md", source_path="/tmp/docs/draft/b.md", status="updated"),
        IngestFileItem(source_file="tickets/RK-123.md", source_path="/tmp/tickets/RK-123.md", status="not_queued"),
        IngestFileItem(source_file="tickets/RK-456.txt", source_path="/tmp/tickets/RK-456.txt", status="compiled"),
    ]

    selected = apply_selection(
        items,
        IngestSelection(
            statuses=["updated", "not_queued"],
            include_globs=["docs/**/*.md", "RK-*.md"],
            exclude_globs=["**/draft/**"],
            search="rk",
        ),
    )

    assert [item.source_file for item in selected] == ["tickets/RK-123.md"]


def test_apply_selection_rejects_path_traversal_patterns():
    try:
        apply_selection(
            [IngestFileItem(source_file="a.md", source_path="/tmp/a.md", status="updated")],
            IngestSelection(include_globs=["../secret.md"]),
        )
    except ValueError as exc:
        assert "Invalid source pattern" in str(exc)
    else:
        raise AssertionError("path traversal pattern should be rejected")


def test_source_identity_returns_relative_path_for_nested_source_root():
    assert source_identity(
        "/project/raw/sources/a/guide.md",
        "/project/raw/sources",
    ) == "a/guide.md"


def test_list_project_source_files_uses_repo_root_when_raw_sources_empty(tmp_path):
    project = tmp_path / "project"
    (project / "docs" / "nested").mkdir(parents=True)
    (project / "raw" / "sources").mkdir(parents=True)
    (project / ".git").mkdir()
    (project / "wiki").mkdir()
    (project / "docs" / "nested" / "guide.md").write_text("guide", encoding="utf-8")
    (project / "README.md").write_text("readme", encoding="utf-8")
    (project / ".git" / "config").write_text("git", encoding="utf-8")
    (project / "wiki" / "index.md").write_text("generated", encoding="utf-8")

    source_root, files = list_project_source_files(str(project))

    assert source_root == project
    assert [path.relative_to(source_root).as_posix() for path in files] == [
        "docs/nested/guide.md",
        "README.md",
    ]


def test_list_project_source_files_prefers_populated_raw_sources(tmp_path):
    project = tmp_path / "project"
    (project / "docs").mkdir(parents=True)
    (project / "raw" / "sources" / "a").mkdir(parents=True)
    (project / "docs" / "outside.md").write_text("outside", encoding="utf-8")
    (project / "raw" / "sources" / "a" / "guide.md").write_text("guide", encoding="utf-8")

    source_root, files = list_project_source_files(str(project))

    assert source_root == project / "raw" / "sources"
    assert [path.relative_to(source_root).as_posix() for path in files] == ["a/guide.md"]


def test_list_project_source_files_prefers_provider_checkout(tmp_path):
    project = tmp_path / "project"
    provider = provider_source_root(str(project))
    (provider / "docs").mkdir(parents=True)
    (project / "raw" / "sources").mkdir(parents=True)
    (provider / "docs" / "guide.md").write_text("guide", encoding="utf-8")
    (project / "raw" / "sources" / "old.md").write_text("old", encoding="utf-8")

    source_root, files = list_project_source_files(str(project))

    assert source_root == provider
    assert [path.relative_to(source_root).as_posix() for path in files] == ["docs/guide.md"]


def test_stage_source_for_ingest_copies_selected_provider_file_and_records_map(tmp_path):
    project = tmp_path / "project"
    provider = provider_source_root(str(project))
    source = provider / "docs" / "guide.md"
    source.parent.mkdir(parents=True)
    source.write_text("guide", encoding="utf-8")

    staged = stage_source_for_ingest(str(project), str(source), str(provider))

    assert staged == project / "raw" / "sources" / "docs" / "guide.md"
    assert staged.read_text(encoding="utf-8") == "guide"
    source_map = load_source_map(str(project))
    assert source_map["docs/guide.md"]["source_path"] == "docs/guide.md"
    assert source_map["docs/guide.md"]["raw_source_path"] == "raw/sources/docs/guide.md"
    assert source_map["docs/guide.md"]["sha256"]


def test_browser_source_roots_split_remote_and_local(tmp_path):
    project = tmp_path / "project"
    provider = provider_source_root(str(project))
    local = local_source_root(str(project))
    (provider / "remote").mkdir(parents=True)
    (local / "uploads").mkdir(parents=True)
    (provider / "remote" / "guide.md").write_text("remote", encoding="utf-8")
    (local / "uploads" / "manual.md").write_text("local", encoding="utf-8")

    remote_files = list_source_files_at_root(browser_source_root(str(project), "remote"))
    local_files = list_source_files_at_root(browser_source_root(str(project), "local"))

    assert [path.relative_to(provider).as_posix() for path in remote_files] == ["remote/guide.md"]
    assert [path.relative_to(local).as_posix() for path in local_files] == ["uploads/manual.md"]
