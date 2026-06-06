"""Tests for knowledge fast-path lookup."""

import os
import tempfile

import pytest

from app.knowledge.fast_lookup import (
    extract_term_from_definition_query,
    fast_lookup,
    is_definition_query,
)


@pytest.fixture
def wiki_dir():
    """Create a temp project dir with wiki/entities and wiki/concepts."""
    with tempfile.TemporaryDirectory() as d:
        entities = os.path.join(d, "wiki", "entities")
        concepts = os.path.join(d, "wiki", "concepts")
        os.makedirs(entities)
        os.makedirs(concepts)

        # Entity page
        with open(os.path.join(entities, "gms.md"), "w") as f:
            f.write(
                "---\n"
                "type: entity\n"
                "title: GMS\n"
                "tags: [android, google]\n"
                "---\n"
                "# GMS\n\n"
                "## 定义\n\n"
                "Google Mobile Services 是 Google 为 Android 提供的移动服务套件。\n\n"
                "## 详细说明\n\n"
                "包括 Play Store、Google Maps 等。\n"
            )

        # Concept page
        with open(os.path.join(concepts, "ota-update.md"), "w") as f:
            f.write(
                "---\n"
                "type: concept\n"
                "title: OTA 升级\n"
                "tags: [android]\n"
                "---\n"
                "# OTA 升级\n\n"
                "## 概述\n\n"
                "Over-The-Air 升级是一种通过无线方式更新设备固件的技术。\n\n"
                "## 流程\n\n"
                "下载 → 校验 → 写入 → 重启\n"
            )

        yield d


class TestIsDefinitionQuery:
    def test_short_term(self):
        assert is_definition_query("GMS") is True

    def test_what_is_pattern(self):
        assert is_definition_query("什么是GMS") is True
        assert is_definition_query("什么是 OTA 升级") is True

    def test_definition_pattern(self):
        assert is_definition_query("GMS的定义") is True

    def test_long_question_not_definition(self):
        assert is_definition_query("GMS 在 Android 12 上升级失败时如何排查？") is False


class TestExtractTermFromQuery:
    def test_what_is(self):
        assert extract_term_from_definition_query("什么是GMS") == "GMS"

    def test_is_what(self):
        assert extract_term_from_definition_query("OTA 升级是什么？") == "OTA 升级"

    def test_plain_term(self):
        assert extract_term_from_definition_query("GMS") == "GMS"


class TestFastLookup:
    def test_slug_match(self, wiki_dir):
        result = fast_lookup(wiki_dir, "gms")
        assert result is not None
        assert result.title == "GMS"
        assert result.matched_by == "slug"
        assert "Google Mobile Services" in result.content

    def test_title_match(self, wiki_dir):
        result = fast_lookup(wiki_dir, "OTA 升级")
        assert result is not None
        assert result.title == "OTA 升级"
        assert result.matched_by == "title"
        assert "Over-The-Air" in result.content

    def test_bm25_match(self, wiki_dir):
        result = fast_lookup(wiki_dir, "Google Mobile Services")
        assert result is not None
        assert "GMS" in result.title or "Google" in result.content

    def test_not_found(self, wiki_dir):
        result = fast_lookup(wiki_dir, "xyznonexistent")
        assert result is None
