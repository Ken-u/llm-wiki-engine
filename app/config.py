"""Application configuration loaded from YAML and environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    projects_dir: str = "./projects"


class AuthConfig(BaseModel):
    jwt_secret: str = "change-me"
    jwt_expire_hours: int = 72
    allow_registration: bool = True


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    api_base: str | None = None
    max_context_size: int = 128000
    ingest_temperature: float = 0.1
    chat_temperature: float = 0.7


class EmbeddingConfig(BaseModel):
    enabled: bool = True
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    api_base: str | None = None
    dimensions: int = 1536


class SearchConfig(BaseModel):
    rrf_k: int = 60
    default_top_k: int = 10
    filename_exact_bonus: float = 200
    phrase_in_title_bonus: float = 50


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)


class Settings(BaseSettings):
    config_path: str = "config.yaml"
    database_url: str = "sqlite+aiosqlite:///./data/engine.db"

    class Config:
        env_file = ".env"
        extra = "ignore"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        return os.environ.get(key, "")
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | None = None) -> AppConfig:
    cfg_path = Path(path or os.environ.get("CONFIG_PATH", "config.yaml"))
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    data = _expand_env(data)

    if os.environ.get("LLM_API_KEY"):
        data.setdefault("llm", {})["api_key"] = os.environ["LLM_API_KEY"]
    if os.environ.get("EMBEDDING_API_KEY"):
        data.setdefault("embedding", {})["api_key"] = os.environ["EMBEDDING_API_KEY"]
    if os.environ.get("JWT_SECRET"):
        data.setdefault("auth", {})["jwt_secret"] = os.environ["JWT_SECRET"]
    if os.environ.get("PROJECTS_DIR"):
        data.setdefault("server", {})["projects_dir"] = os.environ["PROJECTS_DIR"]

    return AppConfig.model_validate(data)


_settings = Settings()
_config: AppConfig | None = None


def get_settings() -> Settings:
    return _settings


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config(_settings.config_path)
    return _config
