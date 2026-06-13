"""Tests for case markdown parser — case-service numbered headings."""

from app.case_index.parser import parse_case_markdown, normalize_heading

CASE_MARKDOWN_MD = """\
---
type: source
title: "Boot Failure Case"
created: 2026-01-01
updated: 2026-01-01
tags: [ticket, case, android]
ticket_id: "2001"
---

# 案例概述

## 1. 问题摘要
设备无法正常启动，卡在开机动画。

## 2. 适用范围
Android 13, reference-device 平台。

## 3. 问题现象
开机后黑屏 30 秒。

## 4. 关键信息
错误码 E001，内核 panic。

## 5. 原因分析
内核 OOM killer 触发。

## 6. 处理过程
调整 low memory killer 参数。

## 7. 最终处理方案
增加系统保留内存至 512MB。

## 8. 结论与规则
低内存设备需预留足够内存。

## 9. 原始对话摘录
工程师：已复现问题。
"""


def test_normalize_heading_strips_number_prefix():
    assert normalize_heading("1. 问题摘要") == "问题摘要"
    assert normalize_heading("6. 处理过程") == "处理过程"
    assert normalize_heading("问题摘要") == "问题摘要"


def test_parse_case_numbered_sections(tmp_path):
    record, chunks = parse_case_markdown(
        CASE_MARKDOWN_MD, "raw/sources/2001.md"
    )
    assert record.case_id == "2001"
    assert record.problem_summary == "设备无法正常启动，卡在开机动画。"
    assert record.scope == "Android 13, reference-device 平台。"
    assert record.symptoms == "开机后黑屏 30 秒。"
    assert record.key_facts == "错误码 E001，内核 panic。"
    assert record.root_cause == "内核 OOM killer 触发。"
    assert record.diagnosis_steps == "调整 low memory killer 参数。"
    assert record.resolution == "增加系统保留内存至 512MB。"
    assert record.rules == "低内存设备需预留足够内存。"
    assert record.dialog_excerpt == "工程师：已复现问题。"
    assert len(chunks) > 0
