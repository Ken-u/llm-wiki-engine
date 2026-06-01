"""Unit tests for user API token helpers and model."""

from app.auth.deps import generate_api_token, hash_api_token
from app.auth.models import UserApiToken


def test_generate_api_token_has_stable_prefix_and_entropy():
    first = generate_api_token()
    second = generate_api_token()

    assert first.startswith("lwu_")
    assert second.startswith("lwu_")
    assert first != second
    assert len(first) > 32


def test_hash_api_token_is_deterministic_and_does_not_store_raw_token():
    raw = "lwu_test_token"

    assert hash_api_token(raw) == hash_api_token(raw)
    assert hash_api_token(raw) != raw


def test_user_api_token_columns_present():
    cols = {c.name for c in UserApiToken.__table__.columns}
    assert {"id", "user_id", "token_hash", "created_at", "last_used_at"}.issubset(cols)

