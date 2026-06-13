"""Tests for case index hybrid search."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.case_index.builder import rebuild_case_index, CASE_INDEX_DIR
from app.case_index.models import CaseManifest
from app.case_index.search import search_cases, read_case, SearchResult


SAMPLE_MD = """\
---
title: Boot Failure Case
ticket_id: "2001"
domain: android
tags:
  - boot
  - kernel
---

# Boot Failure Case

## 问题摘要

设备无法正常启动，卡在开机动画。

## 根因分析

内核 OOM killer 触发，杀死关键系统进程。

## 解决方案

调整 low memory killer 参数，增加系统保留内存。
"""


def _fake_embeddings(texts):
    return [[float(i % 10) / 10.0] * 8 for i, _ in enumerate(texts)]


def _mock_config():
    return SimpleNamespace(
        embedding=SimpleNamespace(model="test-model", dimensions=8, enabled=True)
    )


def _setup_and_build(tmp_path):
    src = tmp_path / "raw" / "sources"
    src.mkdir(parents=True, exist_ok=True)
    (src / "2001.md").write_text(SAMPLE_MD, encoding="utf-8")
    (tmp_path / ".llm-wiki").mkdir(parents=True, exist_ok=True)

    with patch("app.case_index.builder._get_embeddings", new_callable=AsyncMock) as mock_embed, \
         patch("app.case_index.builder.get_config") as mock_cfg:
        mock_embed.side_effect = lambda texts: _fake_embeddings(texts)
        mock_cfg.return_value = _mock_config()
        asyncio.run(rebuild_case_index(str(tmp_path)))


def test_search_cases_returns_structured_results(tmp_path):
    _setup_and_build(tmp_path)

    with patch("app.case_index.search._get_embeddings", new_callable=AsyncMock) as mock_embed:
        mock_embed.side_effect = lambda texts: _fake_embeddings(texts)
        results = asyncio.run(search_cases(str(tmp_path), "boot failure", limit=3))

    assert len(results) > 0
    r = results[0]
    assert r.case_id == "2001"
    assert r.title == "Boot Failure Case"
    assert isinstance(r.matched_sections, list)
    assert r.score > 0


def test_search_cases_no_index_returns_empty(tmp_path):
    (tmp_path / ".llm-wiki").mkdir(parents=True, exist_ok=True)
    results = asyncio.run(search_cases(str(tmp_path), "anything", limit=3))
    assert results == []


def test_read_case_returns_agent_friendly_record(tmp_path):
    _setup_and_build(tmp_path)
    result = read_case(str(tmp_path), "2001")
    assert result is not None
    assert result["source_type"] == "ticket_case_index"
    assert result["case_id"] == "2001"
    assert "sections" in result
    assert "available_sections" in result
    assert "source_path" not in result
    assert "raw_text_hash" not in result
    assert result["sections"]["问题摘要"]


def test_read_case_with_section(tmp_path):
    _setup_and_build(tmp_path)
    result = read_case(str(tmp_path), "2001", section="根因分析")
    assert result is not None
    assert result["source_type"] == "ticket_case_index"
    assert "content" in result
    assert "OOM" in result["content"]


def test_read_case_fuzzy_section_numbered_heading(tmp_path):
    from tests.test_case_index_parser import CASE_MARKDOWN_MD

    src = tmp_path / "raw" / "sources"
    src.mkdir(parents=True, exist_ok=True)
    (src / "3001.md").write_text(CASE_MARKDOWN_MD, encoding="utf-8")
    (tmp_path / ".llm-wiki").mkdir(parents=True, exist_ok=True)
    with patch("app.case_index.builder._get_embeddings", new_callable=AsyncMock) as mock_embed, \
         patch("app.case_index.builder.get_config") as mock_cfg:
        mock_embed.side_effect = lambda texts: _fake_embeddings(texts)
        mock_cfg.return_value = _mock_config()
        asyncio.run(rebuild_case_index(str(tmp_path)))

    result = read_case(str(tmp_path), "2001", section="处理过程")
    assert result is not None
    assert "content" in result
    assert "low memory killer" in result["content"]


def test_read_case_section_not_found_lists_available(tmp_path):
    _setup_and_build(tmp_path)
    result = read_case(str(tmp_path), "2001", section="不存在的章节")
    assert result is not None
    assert "error" in result
    assert "available_sections" in result


def test_read_case_not_found(tmp_path):
    _setup_and_build(tmp_path)
    result = read_case(str(tmp_path), "nonexistent")
    assert result is None


def test_read_case_source_returns_raw_markdown(tmp_path):
    _setup_and_build(tmp_path)
    from app.case_index.search import read_case_source

    result = read_case_source(str(tmp_path), "2001")
    assert result is not None
    assert result["case_id"] == "2001"
    assert "raw_content" in result
    assert "Boot Failure Case" in result["raw_content"]
    assert "source_path" not in result


def test_read_case_source_finds_nested_path(tmp_path):
    from app.case_index.search import read_case_source

    src = tmp_path / "raw" / "sources" / "gms"
    src.mkdir(parents=True)
    (src / "606125.md").write_text(SAMPLE_MD, encoding="utf-8")

    result = read_case_source(str(tmp_path), "606125")
    assert result is not None
    assert "Boot Failure Case" in result["raw_content"]
