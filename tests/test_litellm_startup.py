"""Startup defaults for LiteLLM import behavior."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys


def test_app_defaults_litellm_to_local_model_cost_map(monkeypatch):
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
    sys.modules.pop("app", None)

    import app  # noqa: F401

    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"


def test_app_preserves_explicit_litellm_model_cost_map_setting(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "false")
    sys.modules.pop("app", None)

    app = importlib.import_module("app")

    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "false"
    assert app is not None


def test_app_main_import_does_not_import_litellm():
    script = (
        "import sys; "
        "import app.main; "
        "print('litellm' in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
