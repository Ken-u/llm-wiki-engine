"""Tests for the case index full-rebuild pipeline."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.case_index.builder import rebuild_case_index, CASE_INDEX_DIR
from app.case_index.models import CaseManifest


SAMPLE_MD_1 = """\
---
title: Boot Failure
ticket_id: "1001"
domain: android
tags:
  - boot
---

# Boot Failure

## 问题摘要

设备无法开机。

## 根因分析

内核 panic。

## 解决方案

修复内核模块。
"""

SAMPLE_MD_2 = """\
---
title: WiFi Disconnect
ticket_id: "1002"
domain: connectivity
tags:
  - wifi
---

# WiFi Disconnect

## 问题摘要

WiFi 频繁断连。

## 根因分析

驱动兼容性问题。

## 解决方案

升级 WiFi 驱动。
"""


def _setup_sources(project_dir: Path, files: dict[str, str]):
    src = project_dir / "raw" / "sources"
    src.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (src / name).write_text(content, encoding="utf-8")
    (project_dir / ".llm-wiki").mkdir(parents=True, exist_ok=True)


def _fake_embeddings(texts: list[str]) -> list[list[float]]:
    return [[0.1] * 8 for _ in texts]


def _mock_config():
    return SimpleNamespace(
        embedding=SimpleNamespace(model="test-model", dimensions=8, enabled=True)
    )


def test_rebuild_creates_manifest_and_index(tmp_path):
    _setup_sources(tmp_path, {"1001.md": SAMPLE_MD_1, "1002.md": SAMPLE_MD_2})

    with patch("app.case_index.builder._get_embeddings", new_callable=AsyncMock) as mock_embed, \
         patch("app.case_index.builder.get_config") as mock_cfg:
        mock_embed.return_value = _fake_embeddings([""] * 20)
        mock_embed.side_effect = lambda texts: _fake_embeddings(texts)
        mock_cfg.return_value = _mock_config()
        result = asyncio.run(rebuild_case_index(str(tmp_path)))

    assert result.status == "ready"
    assert result.case_count == 2
    assert result.chunk_count > 0

    index_dir = tmp_path / CASE_INDEX_DIR
    assert (index_dir / "manifest.json").exists()
    assert (index_dir / "cases.jsonl").exists()
    assert (index_dir / "keyword.sqlite").exists()

    manifest = CaseManifest.from_dict(
        json.loads((index_dir / "manifest.json").read_text())
    )
    assert manifest.is_ready
    assert manifest.case_count == 2


def test_rebuild_skips_broken_files(tmp_path):
    broken_md = "This is not valid markdown with frontmatter at all"
    _setup_sources(tmp_path, {"1001.md": SAMPLE_MD_1, "broken.md": broken_md})

    with patch("app.case_index.builder._get_embeddings", new_callable=AsyncMock) as mock_embed, \
         patch("app.case_index.builder.get_config") as mock_cfg:
        mock_embed.side_effect = lambda texts: _fake_embeddings(texts)
        mock_cfg.return_value = _mock_config()
        result = asyncio.run(rebuild_case_index(str(tmp_path)))

    # broken.md still produces a chunk (from body fallback), so both files contribute
    assert result.status == "ready"
    assert result.case_count >= 1


def test_rebuild_embedding_failure_sets_failed(tmp_path):
    _setup_sources(tmp_path, {"1001.md": SAMPLE_MD_1})

    with patch("app.case_index.builder._get_embeddings", new_callable=AsyncMock) as mock_embed, \
         patch("app.case_index.builder.get_config") as mock_cfg:
        mock_embed.side_effect = RuntimeError("API unreachable")
        mock_cfg.return_value = _mock_config()
        result = asyncio.run(rebuild_case_index(str(tmp_path)))

    assert result.status == "failed"
    assert len(result.errors) > 0

    index_dir = tmp_path / CASE_INDEX_DIR
    assert not index_dir.exists() or not (index_dir / "cases.jsonl").exists()


def test_rebuild_atomic_replace(tmp_path):
    """Old index should be replaced only on success."""
    _setup_sources(tmp_path, {"1001.md": SAMPLE_MD_1})
    old_index = tmp_path / CASE_INDEX_DIR
    old_index.mkdir(parents=True, exist_ok=True)
    (old_index / "manifest.json").write_text('{"status": "ready", "case_count": 99}')

    with patch("app.case_index.builder._get_embeddings", new_callable=AsyncMock) as mock_embed, \
         patch("app.case_index.builder.get_config") as mock_cfg:
        mock_embed.side_effect = lambda texts: _fake_embeddings(texts)
        mock_cfg.return_value = _mock_config()
        result = asyncio.run(rebuild_case_index(str(tmp_path)))

    assert result.status == "ready"
    manifest = json.loads((old_index / "manifest.json").read_text())
    assert manifest["case_count"] != 99
