from app.config import normalize_litellm_api_base


def test_normalize_litellm_api_base_appends_v1():
    assert normalize_litellm_api_base("http://example.com:3000") == "http://example.com:3000/v1"


def test_normalize_litellm_api_base_keeps_existing_v1():
    assert normalize_litellm_api_base("http://example.com:3000/v1") == "http://example.com:3000/v1"
    assert normalize_litellm_api_base("http://example.com:3000/v1/") == "http://example.com:3000/v1"


def test_normalize_litellm_api_base_none():
    assert normalize_litellm_api_base(None) is None
    assert normalize_litellm_api_base("") is None
