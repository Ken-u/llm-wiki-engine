"""Unit tests for git sync service."""

from app.projects.git_sync import _inject_auth


def test_inject_auth_with_token():
    url = "https://git.example.com/org/repo.git"
    result = _inject_auth(url, "my_token", username="bot")
    assert "bot:my_token@git.example.com" in result
    assert result.startswith("https://")


def test_inject_auth_without_token():
    url = "https://git.example.com/org/repo.git"
    result = _inject_auth(url, "")
    assert result == url


def test_inject_auth_special_chars():
    url = "https://git.example.com/org/repo.git"
    result = _inject_auth(url, "p@ss/w0rd!", username="us er")
    assert "us%20er" in result
    assert "p%40ss%2Fw0rd%21" in result


def test_inject_auth_with_port():
    url = "https://git.example.com:8443/repo.git"
    result = _inject_auth(url, "tok", username="u")
    assert ":8443" in result
    assert "u:tok@" in result


def test_inject_auth_non_https_unchanged():
    url = "git@github.com:org/repo.git"
    result = _inject_auth(url, "token")
    assert result == url


def test_inject_auth_default_username():
    url = "https://git.example.com/repo.git"
    result = _inject_auth(url, "token123")
    assert "oauth2:token123@" in result
