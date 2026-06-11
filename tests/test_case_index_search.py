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


def test_read_case_returns_record(tmp_path):
    _setup_and_build(tmp_path)
    result = read_case(str(tmp_path), "2001")
    assert result is not None
    assert result["case_id"] == "2001"
    assert "problem_summary" in result


def test_read_case_with_section(tmp_path):
    _setup_and_build(tmp_path)
    result = read_case(str(tmp_path), "2001", section="根因分析")
    assert result is not None
    assert "content" in result


def test_read_case_not_found(tmp_path):
    _setup_and_build(tmp_path)
    result = read_case(str(tmp_path), "nonexistent")
    assert result is None
