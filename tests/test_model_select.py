"""Tests for virtual model name helpers."""

from app.llm.model_select import (
    FAST_MODEL_SUFFIX,
    fast_virtual_model_id,
    parse_virtual_model,
    resolve_api_use_fast_model,
)


def test_parse_virtual_model_fast_suffix():
    base, use_fast = parse_virtual_model("my-wiki-fast")
    assert base == "my-wiki"
    assert use_fast is True


def test_parse_virtual_model_default():
    base, use_fast = parse_virtual_model("my-wiki")
    assert base == "my-wiki"
    assert use_fast is False


def test_fast_virtual_model_id():
    assert fast_virtual_model_id("my-wiki") == f"my-wiki{FAST_MODEL_SUFFIX}"


def test_resolve_api_use_fast_model_from_suffix():
    base, use_fast = resolve_api_use_fast_model(model="my-wiki-fast")
    assert base == "my-wiki"
    assert use_fast is True


def test_resolve_api_use_fast_model_default_fast_without_model(monkeypatch):
    monkeypatch.setattr("app.llm.model_select.fast_model_available", lambda: True)
    base, use_fast = resolve_api_use_fast_model(
        model="",
        default_fast_without_model=True,
    )
    assert base == ""
    assert use_fast is True
