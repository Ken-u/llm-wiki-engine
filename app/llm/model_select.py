"""Helpers for selecting default vs fast LLM via virtual model names."""

from __future__ import annotations

FAST_MODEL_SUFFIX = "-fast"


def parse_virtual_model(model: str) -> tuple[str, bool]:
    """Split ``{base_model}-fast`` into base model id and fast flag."""
    model = (model or "").strip()
    if model.endswith(FAST_MODEL_SUFFIX):
        base = model[: -len(FAST_MODEL_SUFFIX)]
        if base:
            return base, True
    return model, False


def fast_virtual_model_id(base_model_id: str) -> str:
    return f"{base_model_id}{FAST_MODEL_SUFFIX}"


def fast_model_available() -> bool:
    from app.config import get_config

    return bool(get_config().llm.fast_model)


def resolve_api_use_fast_model(
    *,
    model: str | None = None,
    use_fast_model: bool = False,
    default_fast_without_model: bool = False,
) -> tuple[str, bool]:
    """Resolve base model id and whether to use the fast LLM profile."""
    if model and model.strip():
        base_model, fast_from_suffix = parse_virtual_model(model)
        return base_model, use_fast_model or fast_from_suffix
    if default_fast_without_model and fast_model_available():
        return "", True
    return "", use_fast_model
