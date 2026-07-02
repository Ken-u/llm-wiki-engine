"""Runtime status checks."""

from __future__ import annotations

from pathlib import Path

import lancedb

from app.case_index.builder import load_manifest
from app.embedding.service import TABLE_NAME, _lancedb_path
from app.runtime.config import RuntimeSettings
from app.runtime.hooks import get_hook_results


async def build_status(settings: RuntimeSettings) -> dict:
    knowledge = Path(settings.knowledge.path)
    cases = Path(settings.case_library.path)

    vector_status: dict = {
        "exists": False,
        "table_exists": False,
        "table_vector_dimension": None,
        "configured_dimensions": settings.embedding.dimensions,
        "error": "",
    }
    try:
        db_path = _lancedb_path(str(knowledge))
        vector_status["exists"] = Path(db_path).exists()
        if vector_status["exists"]:
            db = await lancedb.connect_async(db_path)
            vector_status["table_exists"] = TABLE_NAME in await db.table_names()
            if vector_status["table_exists"]:
                table = await db.open_table(TABLE_NAME)
                schema_attr = table.schema
                schema = await schema_attr() if callable(schema_attr) else schema_attr
                vector_field = schema.field("vector")
                vector_type = vector_field.type
                vector_status["table_vector_dimension"] = getattr(vector_type, "list_size", None)
    except Exception as exc:
        vector_status["error"] = str(exc)

    manifest = load_manifest(str(cases)) if settings.case_library.enabled else None

    return {
        "runtime": {"version": "0.1.0"},
        "knowledge": {
            "name": settings.knowledge.name,
            "path": settings.knowledge.path,
            "path_exists": knowledge.exists(),
            "wiki_exists": (knowledge / "wiki").exists(),
            "raw_sources_exists": (knowledge / "raw" / "sources").exists(),
            "vector_index": vector_status,
            "model_name": settings.knowledge.model_name,
        },
        "case_library": {
            "enabled": settings.case_library.enabled,
            "name": settings.case_library.name,
            "path": settings.case_library.path,
            "path_exists": cases.exists(),
            "manifest": manifest.to_dict() if manifest else None,
            "status": manifest.status if manifest else "missing",
        },
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "api_base": settings.llm.api_base,
            "api_key_configured": bool(settings.llm.api_key),
        },
        "embedding": {
            "enabled": settings.embedding.enabled,
            "provider": settings.embedding.provider,
            "model": settings.embedding.model,
            "api_base": settings.embedding.api_base,
            "api_key_configured": bool(settings.embedding.api_key),
        },
        "hooks": get_hook_results(),
    }
