"""LLM Wiki Knowledge Compilation Engine."""

from __future__ import annotations

import os

# Avoid a startup-time network fetch when LiteLLM is imported by routers.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
