"""Runtime project adapters and source tree helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.runtime.config import RuntimeSettings, get_runtime_config
from app.wiki.frontmatter import parse_frontmatter


class ProjectLike(Protocol):
    id: str
    name: str
    disk_path: str
    project_type: str
    ticket_project_id: str | None


@dataclass
class RuntimeProject:
    id: str
    name: str
    disk_path: str
    project_type: str = "knowledge_base"
    ticket_project_id: str | None = None


def get_knowledge_project(settings: RuntimeSettings | None = None) -> RuntimeProject:
    cfg = settings or get_runtime_config()
    return RuntimeProject(
        id="knowledge",
        name=cfg.knowledge.name,
        disk_path=cfg.knowledge.path,
        project_type="knowledge_base",
        ticket_project_id="cases" if cfg.case_library.enabled else None,
    )


def get_case_project(settings: RuntimeSettings | None = None) -> RuntimeProject | None:
    cfg = settings or get_runtime_config()
    if not cfg.case_library.enabled:
        return None
    return RuntimeProject(
        id="cases",
        name=cfg.case_library.name,
        disk_path=cfg.case_library.path,
        project_type="case_library",
    )


def safe_join(base: str, rel_path: str) -> Path | None:
    if ".." in rel_path or rel_path.startswith("/") or rel_path.startswith("\\"):
        return None
    root = Path(base).resolve()
    candidate = (root / rel_path).resolve()
    if root != candidate and root not in candidate.parents:
        return None
    return candidate


def build_wiki_tree(project: ProjectLike) -> list[dict]:
    base = Path(project.disk_path)
    wiki_dir = base / "wiki"
    if not wiki_dir.exists():
        return []

    def walk(path: Path) -> list[dict]:
        items: list[dict] = []
        for entry in sorted(path.iterdir()):
            if entry.name.startswith("."):
                continue
            rel = entry.relative_to(base).as_posix()
            if entry.is_dir():
                items.append({
                    "name": entry.name,
                    "path": rel,
                    "type": "directory",
                    "children": walk(entry),
                })
            elif entry.suffix == ".md":
                title = ""
                try:
                    meta, _ = parse_frontmatter(entry.read_text(encoding="utf-8", errors="replace"))
                    title = meta.title
                except Exception:
                    title = ""
                items.append({
                    "name": entry.name,
                    "path": rel,
                    "type": "file",
                    "title": title,
                })
        return items

    return walk(wiki_dir)
