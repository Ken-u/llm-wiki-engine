"""Runtime configuration for the local single-project inference app."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from app import config as app_config
from app.config import (
    AppConfig,
    EmbeddingConfig,
    LLMConfig,
    SearchConfig,
    ServerConfig,
)


class RuntimeServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8012
    open_browser: bool = True
    api_key: str = ""


class RuntimeKnowledgeConfig(BaseModel):
    name: str = "Local Knowledge"
    path: str = "./data/knowledge"
    model_name: str = "local-wiki"
    system_prompt: str = (
        "你是知识库问答助手。只能基于知识库和案例库内容回答。"
        "如果没有找到依据，请明确说明未找到相关信息。"
    )


class RuntimeCaseLibraryConfig(BaseModel):
    enabled: bool = True
    name: str = "Case Library"
    path: str = "./data/cases"


class RuntimeBehaviorConfig(BaseModel):
    mode: Literal["auto", "agent", "fast", "rag"] = "auto"
    max_tool_calls: int = Field(default=20, ge=1, le=200)
    debug_result_limit: int = Field(default=2000, ge=500, le=50000)


class HookWaitForConfig(BaseModel):
    url: str = ""
    timeout_seconds: int = Field(default=60, ge=1)


class HookCommandConfig(BaseModel):
    windows: list[str] = Field(default_factory=list)
    darwin: list[str] = Field(default_factory=list)
    linux: list[str] = Field(default_factory=list)


class HookScriptConfig(BaseModel):
    name: str
    command: HookCommandConfig
    wait_for: HookWaitForConfig | None = None


class RuntimeHooksConfig(BaseModel):
    enabled: bool = False
    run_on_startup: bool = True
    run_before_server: bool = True
    stop_on_failure: bool = True
    timeout_seconds: int = Field(default=120, ge=1)
    scripts: list[HookScriptConfig] = Field(default_factory=list)


class RuntimeSettings(BaseModel):
    server: RuntimeServerConfig = Field(default_factory=RuntimeServerConfig)
    knowledge: RuntimeKnowledgeConfig = Field(default_factory=RuntimeKnowledgeConfig)
    case_library: RuntimeCaseLibraryConfig = Field(default_factory=RuntimeCaseLibraryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    runtime: RuntimeBehaviorConfig = Field(default_factory=RuntimeBehaviorConfig)
    hooks: RuntimeHooksConfig = Field(default_factory=RuntimeHooksConfig)


_settings: RuntimeSettings | None = None
_config_path: Path | None = None


def _expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _default_config_dict() -> dict[str, Any]:
    return RuntimeSettings().model_dump()


def ensure_config_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_default_config_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _resolve_path(config_dir: Path, raw_path: str) -> str:
    p = Path(raw_path).expanduser()
    if not p.is_absolute():
        p = config_dir / p
    return str(p.resolve())


def _resolve_paths(settings: RuntimeSettings, config_dir: Path) -> RuntimeSettings:
    settings.knowledge.path = _resolve_path(config_dir, settings.knowledge.path)
    settings.case_library.path = _resolve_path(config_dir, settings.case_library.path)
    return settings


def _install_shared_app_config(settings: RuntimeSettings) -> None:
    app_config._config = AppConfig(
        server=ServerConfig(
            host=settings.server.host,
            port=settings.server.port,
            projects_dir=str(Path(settings.knowledge.path).parent),
        ),
        llm=settings.llm,
        embedding=settings.embedding,
        search=settings.search,
    )


def load_runtime_config(path: str | Path = "config.yaml", *, create: bool = True) -> RuntimeSettings:
    global _settings, _config_path

    cfg_path = Path(path).expanduser().resolve()
    if create:
        ensure_config_file(cfg_path)

    data: dict[str, Any] = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    data = _expand_env(data)

    settings = RuntimeSettings.model_validate(data)
    settings = _resolve_paths(settings, cfg_path.parent)

    _settings = settings
    _config_path = cfg_path
    _install_shared_app_config(settings)
    return settings


def get_runtime_config() -> RuntimeSettings:
    if _settings is None:
        return load_runtime_config(os.environ.get("RUNTIME_CONFIG", "config.yaml"))
    return _settings


def get_runtime_config_path() -> Path | None:
    return _config_path
