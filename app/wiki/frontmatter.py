"""YAML frontmatter parsing for wiki pages."""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class PageMeta:
    type: str = ""
    title: str = ""
    tags: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    created: str = ""
    updated: str = ""
    raw: dict = field(default_factory=dict)


def parse_frontmatter(content: str) -> tuple[PageMeta, str]:
    """Parse YAML frontmatter from markdown content. Returns (meta, body)."""
    stripped = content.strip()
    if not stripped.startswith("---"):
        return PageMeta(), content

    end = stripped.find("---", 3)
    if end == -1:
        return PageMeta(), content

    fm_text = stripped[3:end].strip()
    body = stripped[end + 3:].strip()

    try:
        data = yaml.safe_load(fm_text)
        if not isinstance(data, dict):
            return PageMeta(), content
    except yaml.YAMLError:
        return PageMeta(), content

    meta = PageMeta(
        type=str(data.get("type", "")),
        title=str(data.get("title", "")),
        tags=_to_list(data.get("tags")),
        related=_to_list(data.get("related")),
        sources=_to_list(data.get("sources")),
        created=str(data.get("created", "")),
        updated=str(data.get("updated", "")),
        raw=data,
    )
    return meta, body


def _to_list(val) -> list[str]:
    if isinstance(val, list):
        return [str(v) for v in val]
    if isinstance(val, str):
        return [val] if val else []
    return []
