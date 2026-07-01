"""Application configuration loaded from YAML, env vars, and database.

Load priority (later wins):
  1. config.yaml (base defaults)
  2. Environment variables (deployment overrides)
  3. Database system_settings table (admin UI changes, highest priority)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


def normalize_litellm_api_base(api_base: str | None) -> str | None:
    """Ensure OpenAI-compatible api_base ends with /v1 for LiteLLM."""
    if not api_base:
        return None
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


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
    context_compress_threshold: float = 0.85
    context_compress_target: float = 0.65
    timeout: int = 120
    ingest_temperature: float = 0.1
    chat_temperature: float = 0.7
    stream: bool = False


class EmbeddingConfig(BaseModel):
    enabled: bool = True
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    api_key: str = ""
    api_base: str | None = None
    dimensions: int | None = None


class SearchConfig(BaseModel):
    rrf_k: int = 60
    default_top_k: int = 10
    filename_exact_bonus: float = 200
    phrase_in_title_bonus: float = 50


class FeedbackModelConfig(BaseModel):
    """LLM config for a feedback pipeline agent. None = inherit from main llm."""
    model: str | None = None
    provider: str | None = None
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 120


class FeedbackConfig(BaseModel):
    enabled: bool = True
    evaluator: FeedbackModelConfig = Field(default_factory=FeedbackModelConfig)
    compiler: FeedbackModelConfig = Field(default_factory=FeedbackModelConfig)
    max_revisions: int = 5
    dedup_window_minutes: int = 60


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)


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


def _load_yaml(path: str | None = None) -> dict[str, Any]:
    """Load config.yaml + env var overrides (layers 1+2)."""
    cfg_path = Path(path or os.environ.get("CONFIG_PATH", "config.yaml"))
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    data = _expand_env(data)

    # Env vars override yaml
    if os.environ.get("LLM_API_KEY"):
        data.setdefault("llm", {})["api_key"] = os.environ["LLM_API_KEY"]
    if os.environ.get("EMBEDDING_API_KEY"):
        data.setdefault("embedding", {})["api_key"] = os.environ["EMBEDDING_API_KEY"]
    if os.environ.get("JWT_SECRET"):
        data.setdefault("auth", {})["jwt_secret"] = os.environ["JWT_SECRET"]
    if os.environ.get("PROJECTS_DIR"):
        data.setdefault("server", {})["projects_dir"] = os.environ["PROJECTS_DIR"]

    return data


def load_config(path: str | None = None) -> AppConfig:
    data = _load_yaml(path)
    return AppConfig.model_validate(data)


# ── Singleton ──

_settings = Settings()
_config: AppConfig | None = None


def get_settings() -> Settings:
    return _settings


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config(_settings.config_path)
    return _config


def _reload_config() -> AppConfig:
    """Force reload from yaml+env, then overlay DB settings."""
    global _config
    _config = load_config(_settings.config_path)
    return _config


# ── DB-backed settings (layer 3) ──

def _db_url_sync() -> str:
    """Convert async sqlite URL to sync for simple reads."""
    return _settings.database_url.replace("sqlite+aiosqlite", "sqlite")


def load_db_overrides() -> None:
    """Read system_settings from DB and overlay onto in-memory config.

    Called once at startup after init_db, and after each admin save.
    Uses a synchronous connection since it only runs at known safe points.
    """
    import sqlalchemy
    global _config

    cfg = get_config()
    sync_url = _db_url_sync()

    try:
        engine = sqlalchemy.create_engine(
            sync_url,
            connect_args={"timeout": 30} if sync_url.startswith("sqlite") else {},
        )
        with engine.connect() as conn:
            if sync_url.startswith("sqlite"):
                conn.execute(sqlalchemy.text("PRAGMA busy_timeout=30000"))
            # Check table exists
            inspector = sqlalchemy.inspect(engine)
            if "system_settings" not in inspector.get_table_names():
                engine.dispose()
                return

            rows = conn.execute(sqlalchemy.text(
                "SELECT section, data FROM system_settings"
            )).fetchall()
        engine.dispose()
    except Exception:
        logger.debug("system_settings table not available yet, skipping DB overrides")
        return

    for section, data_json in rows:
        try:
            values = json.loads(data_json)
        except (json.JSONDecodeError, TypeError):
            continue

        sub = getattr(cfg, section, None)
        if sub is None:
            continue
        _apply_overrides(sub, values)

    logger.info("Loaded admin settings from DB for sections: %s", [r[0] for r in rows])


def _apply_overrides(target: Any, values: dict) -> None:
    """Recursively apply override values to a Pydantic model instance."""
    for k, v in values.items():
        if not hasattr(target, k) or v is None:
            continue
        current = getattr(target, k)
        if isinstance(v, dict) and hasattr(current, "__fields__"):
            _apply_overrides(current, v)
        else:
            setattr(target, k, v)


async def save_db_settings(section: str, values: dict[str, Any]) -> None:
    """Persist admin settings to the system_settings table, then refresh in-memory config."""
    global _config
    from app.database import async_session

    async with async_session() as db:
        from sqlalchemy import text

        existing = (await db.execute(
            text("SELECT data FROM system_settings WHERE section = :s"),
            {"s": section},
        )).scalar_one_or_none()

        if existing:
            current = json.loads(existing)
            current.update(values)
            await db.execute(
                text("UPDATE system_settings SET data = :d WHERE section = :s"),
                {"d": json.dumps(current, ensure_ascii=False), "s": section},
            )
        else:
            await db.execute(
                text("INSERT INTO system_settings (section, data) VALUES (:s, :d)"),
                {"s": section, "d": json.dumps(values, ensure_ascii=False)},
            )

        await db.commit()

    # Refresh: reload yaml+env base, then overlay all DB settings
    _reload_config()
    load_db_overrides()
