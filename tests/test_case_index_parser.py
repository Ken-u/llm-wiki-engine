"""Tests for case index data models and parser."""

import json
from datetime import datetime, timezone

from app.case_index.models import CaseRecord, CaseChunk, CaseManifest


def test_case_record_to_dict():
    rec = CaseRecord(
        case_id="12345",
        ticket_id="12345",
        title="GMS 启动失败",
        domain="android",
        tags=["gms", "boot"],
        source_path="raw/sources/12345.md",
        updated_at="2026-06-01",
        problem_summary="设备开机后 GMS 无法启动",
        root_cause="签名不匹配",
        resolution="重新签名 APK",
        diagnosis_steps="检查 logcat → 确认签名",
        affected_modules=["GmsCore"],
        raw_text_hash="abc123",
    )
    d = rec.to_dict()
    assert d["case_id"] == "12345"
    assert d["domain"] == "android"
    assert d["tags"] == ["gms", "boot"]
    assert d["affected_modules"] == ["GmsCore"]

    restored = CaseRecord.from_dict(d)
    assert restored.case_id == rec.case_id
    assert restored.resolution == rec.resolution


def test_case_chunk_fields():
    chunk = CaseChunk(
        chunk_id="12345#0",
        case_id="12345",
        source_path="raw/sources/12345.md",
        section="问题摘要",
        heading_path="## 问题摘要",
        chunk_text="设备开机后 GMS 无法启动，logcat 显示签名校验失败。",
    )
    assert chunk.chunk_id == "12345#0"
    assert chunk.section == "问题摘要"


def test_manifest_ready():
    m = CaseManifest(
        status="ready",
        built_at=datetime(2026, 6, 11, tzinfo=timezone.utc).isoformat(),
        source_count=10,
        case_count=8,
        chunk_count=50,
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        errors=[],
        index_version=1,
    )
    assert m.is_ready
    d = m.to_dict()
    assert d["status"] == "ready"
    restored = CaseManifest.from_dict(d)
    assert restored.is_ready
    assert restored.case_count == 8


def test_manifest_failed():
    m = CaseManifest(
        status="failed",
        built_at=datetime(2026, 6, 11, tzinfo=timezone.utc).isoformat(),
        source_count=10,
        case_count=0,
        chunk_count=0,
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        errors=["Embedding API failed"],
        index_version=1,
    )
    assert not m.is_ready
