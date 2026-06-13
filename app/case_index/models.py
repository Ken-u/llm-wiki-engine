"""Data models for the dedicated case index."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class CaseRecord:
    case_id: str
    ticket_id: str
    title: str
    domain: str
    tags: list[str]
    source_path: str
    updated_at: str
    problem_summary: str
    root_cause: str
    resolution: str
    diagnosis_steps: str
    scope: str = ""
    symptoms: str = ""
    key_facts: str = ""
    rules: str = ""
    dialog_excerpt: str = ""
    affected_modules: list[str] = field(default_factory=list)
    raw_text_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaseRecord:
        kwargs: dict[str, Any] = {}
        for k in cls.__dataclass_fields__:
            if k in d:
                kwargs[k] = d[k]
            elif k in ("tags", "affected_modules"):
                kwargs[k] = []
            else:
                kwargs[k] = ""
        return cls(**kwargs)


@dataclass
class CaseChunk:
    chunk_id: str
    case_id: str
    source_path: str
    section: str
    heading_path: str
    chunk_text: str
    vector: list[float] = field(default_factory=list)


@dataclass
class CaseManifest:
    status: str  # "ready" or "failed"
    built_at: str
    source_count: int
    case_count: int
    chunk_count: int
    embedding_model: str
    embedding_dimensions: int
    errors: list[str]
    index_version: int = 1

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CaseManifest:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})
