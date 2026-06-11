"""Tests for case index agent tool integration."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.agents.tools import ToolContext, execute_tool, get_tool_definitions
from app.case_index.builder import CASE_INDEX_DIR
from app.case_index.models import CaseManifest


def _make_project(tmp_path, project_id="case-proj"):
    return SimpleNamespace(
        id=project_id,
        name="Case Library",
        disk_path=str(tmp_path),
    )


def _write_manifest(project_dir, status="ready"):
    idx = Path(project_dir) / CASE_INDEX_DIR
    idx.mkdir(parents=True, exist_ok=True)
    m = CaseManifest(
        status=status,
        built_at="2026-06-11T00:00:00Z",
        source_count=5,
        case_count=3,
        chunk_count=20,
        embedding_model="test",
        embedding_dimensions=8,
        errors=[],
    )
    (idx / "manifest.json").write_text(
        json.dumps(m.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )


def _write_cases_jsonl(project_dir):
    idx = Path(project_dir) / CASE_INDEX_DIR
    idx.mkdir(parents=True, exist_ok=True)
    case = {
        "case_id": "5001",
        "ticket_id": "5001",
        "title": "Test Case",
        "domain": "test",
        "tags": [],
        "source_path": "raw/sources/5001.md",
        "updated_at": "",
        "problem_summary": "Test problem",
        "root_cause": "Test root cause",
        "resolution": "Test resolution",
        "diagnosis_steps": "",
        "affected_modules": [],
        "raw_text_hash": "abc",
    }
    (idx / "cases.jsonl").write_text(
        json.dumps(case, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_tool_definitions_include_read_ticket_case(tmp_path):
    proj = _make_project(tmp_path)
    ctx = ToolContext(main_projects=[], ticket_project=proj)
    defs = get_tool_definitions(ctx)
    names = [d["function"]["name"] for d in defs]
    assert "search_ticket_cases" in names
    assert "read_ticket_case" in names
    assert "read_ticket_page" not in names


def test_tool_definitions_no_ticket_project():
    ctx = ToolContext(main_projects=[], ticket_project=None)
    defs = get_tool_definitions(ctx)
    names = [d["function"]["name"] for d in defs]
    assert "search_ticket_cases" not in names
    assert "read_ticket_case" not in names


def test_search_ticket_cases_no_index(tmp_path):
    proj = _make_project(tmp_path)
    ctx = ToolContext(main_projects=[], ticket_project=proj)
    result = asyncio.run(execute_tool(
        "search_ticket_cases", {"query": "boot failure"}, ctx
    ))
    assert "error" in result
    assert "not built" in result["error"].lower() or "not ready" in result["error"].lower()


def test_search_ticket_cases_with_index(tmp_path):
    proj = _make_project(tmp_path)
    _write_manifest(str(tmp_path), status="ready")
    _write_cases_jsonl(str(tmp_path))
    ctx = ToolContext(main_projects=[], ticket_project=proj)

    with patch("app.agents.tools.search_cases", new_callable=AsyncMock) as mock_search:
        from app.case_index.search import SearchResult, MatchedSection
        mock_search.return_value = [
            SearchResult(
                case_id="5001",
                title="Test Case",
                domain="test",
                problem_summary="Test problem",
                root_cause="Test root cause",
                resolution="Test resolution",
                matched_sections=[MatchedSection(section="问题摘要", snippet="Test")],
                score=0.85,
            )
        ]
        result = asyncio.run(execute_tool(
            "search_ticket_cases", {"query": "test", "limit": 3}, ctx
        ))

    assert result["source_type"] == "ticket_case_index"
    assert len(result["results"]) == 1
    assert result["results"][0]["case_id"] == "5001"
    assert "usage_hint" in result


def test_read_ticket_case_found(tmp_path):
    proj = _make_project(tmp_path)
    _write_manifest(str(tmp_path))
    _write_cases_jsonl(str(tmp_path))
    ctx = ToolContext(main_projects=[], ticket_project=proj)

    with patch("app.agents.tools.read_case") as mock_read:
        mock_read.return_value = {
            "case_id": "5001",
            "title": "Test Case",
            "problem_summary": "Test problem",
        }
        result = asyncio.run(execute_tool(
            "read_ticket_case", {"case_id": "5001"}, ctx
        ))

    assert result["case_id"] == "5001"


def test_read_ticket_case_not_found(tmp_path):
    proj = _make_project(tmp_path)
    _write_manifest(str(tmp_path))
    _write_cases_jsonl(str(tmp_path))
    ctx = ToolContext(main_projects=[], ticket_project=proj)

    with patch("app.agents.tools.read_case") as mock_read:
        mock_read.return_value = None
        result = asyncio.run(execute_tool(
            "read_ticket_case", {"case_id": "9999"}, ctx
        ))

    assert "error" in result


# ── Prompt policy tests ──

from app.agents.orchestrator import _build_system_prompt


def test_system_prompt_includes_case_search_policy():
    prompt = _build_system_prompt("", has_ticket=True)
    assert "search_ticket_cases" in prompt
    assert "read_ticket_case" in prompt
    assert "read_ticket_page" not in prompt
    assert "case_id" in prompt


def test_system_prompt_no_ticket_has_no_case_policy():
    prompt = _build_system_prompt("", has_ticket=False)
    assert "search_ticket_cases" not in prompt
