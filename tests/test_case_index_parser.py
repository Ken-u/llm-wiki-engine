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


# ── Parser tests ──

from app.case_index.parser import parse_case_markdown, generate_case_id


SAMPLE_CASE_MD = """\
---
title: GMS 启动失败
ticket_id: "12345"
domain: android
tags:
  - gms
  - boot-failure
---

# GMS 启动失败

## 案例概述

某客户设备开机后 GMS 无法正常启动，影响所有 Google 服务。

## 问题摘要

设备开机后 logcat 显示 GMS 签名校验失败，所有 Google 应用无法使用。

## 根因分析

系统分区的 GmsCore APK 签名与 OTA 包中预期签名不匹配，
导致 PackageManager 拒绝加载。

## 处理过程

1. 检查 logcat 发现签名校验异常
2. 对比 APK 证书指纹
3. 确认 OTA 打包流程未更新签名

## 解决方案

重新使用正确密钥签名 GmsCore APK，发布修正 OTA。

## 影响范围

涉及所有使用该 OTA 包的设备型号。
"""


def test_parse_case_markdown_extracts_fields():
    record, chunks = parse_case_markdown(SAMPLE_CASE_MD, "raw/sources/12345.md")
    assert record.case_id == "12345"
    assert record.ticket_id == "12345"
    assert record.title == "GMS 启动失败"
    assert record.domain == "android"
    assert "gms" in record.tags
    assert "签名" in record.root_cause
    assert "重新" in record.resolution
    assert len(chunks) > 0
    assert all(c.case_id == "12345" for c in chunks)


def test_parse_case_markdown_no_ticket_id_generates_from_path():
    md = """\
---
title: 未知问题
domain: general
---

# 未知问题

## 问题摘要

测试描述
"""
    record, chunks = parse_case_markdown(md, "raw/sources/misc/unknown-issue.md")
    assert record.case_id == "unknown-issue"
    assert record.ticket_id == ""


def test_parse_case_synonym_headings():
    md = """\
---
title: 同义标题测试
ticket_id: "99"
---

# 同义标题测试

## 原因分析

这是根因内容。

## 修复方案

这是解决方案内容。
"""
    record, _ = parse_case_markdown(md, "raw/sources/99.md")
    assert "这是根因内容" in record.root_cause
    assert "这是解决方案内容" in record.resolution


def test_generate_case_id_from_path():
    assert generate_case_id("raw/sources/12345.md") == "12345"
    assert generate_case_id("raw/sources/domain/my-case.md") == "my-case"
    assert generate_case_id("raw/sources/a/b/deep-nested.md") == "deep-nested"
