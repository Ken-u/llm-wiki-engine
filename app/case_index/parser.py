"""Parse case markdown files into CaseRecord + CaseChunk lists."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.case_index.models import CaseRecord, CaseChunk
from app.wiki.frontmatter import parse_frontmatter

_SYNONYMS: dict[str, list[str]] = {
    "overview": ["案例概述", "概述", "overview"],
    "problem_summary": ["问题摘要", "问题描述", "问题概述", "故障描述", "故障现象", "problem summary"],
    "root_cause": ["根因分析", "原因分析", "根因", "原因", "root cause"],
    "diagnosis_steps": ["处理过程", "排查过程", "诊断步骤", "排查步骤", "处理步骤", "diagnosis"],
    "resolution": ["解决方案", "修复方案", "解决办法", "修复措施", "处理方案", "resolution"],
    "impact": ["影响范围", "影响面", "影响", "impact"],
    "logs": ["相关日志", "日志", "logs", "log"],
    "evidence": ["证据", "evidence"],
}

SECTION_MAP: dict[str, str] = {}
for _field_name, _titles in _SYNONYMS.items():
    for _t in _titles:
        SECTION_MAP[_t.lower()] = _field_name

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)", re.MULTILINE)

MAX_FIELD_CHARS = 500
MAX_CHUNK_CHARS = 800


def generate_case_id(source_path: str) -> str:
    return Path(source_path).stem


def _split_sections(body: str) -> list[tuple[str, str, str]]:
    """Split markdown body into (heading_text, normalized_field, content) tuples."""
    sections: list[tuple[str, str, str]] = []
    matches = list(_HEADING_RE.finditer(body))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        content = body[start:end].strip()
        normalized = SECTION_MAP.get(heading.lower(), "")
        sections.append((heading, normalized, content))

    return sections


def parse_case_markdown(
    text: str, source_path: str
) -> tuple[CaseRecord, list[CaseChunk]]:
    """Parse a case markdown file into a CaseRecord and CaseChunk list."""
    meta, body = parse_frontmatter(text)

    ticket_id = str(meta.raw.get("ticket_id", "")).strip().strip('"').strip("'")
    case_id = ticket_id if ticket_id else generate_case_id(source_path)
    title = meta.title or case_id
    domain = str(meta.raw.get("domain", "")).strip()
    tags = meta.tags
    updated_at = meta.updated or meta.created or ""

    sections = _split_sections(body)

    field_values: dict[str, str] = {
        "problem_summary": "",
        "root_cause": "",
        "resolution": "",
        "diagnosis_steps": "",
    }

    for _heading, field_name, content in sections:
        if field_name in field_values:
            field_values[field_name] = content[:MAX_FIELD_CHARS]

    modules_raw = meta.raw.get("affected_modules", [])
    affected_modules: list[str] = (
        [str(m) for m in modules_raw] if isinstance(modules_raw, list) else []
    )

    raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    record = CaseRecord(
        case_id=case_id,
        ticket_id=ticket_id,
        title=title,
        domain=domain,
        tags=tags,
        source_path=source_path,
        updated_at=updated_at,
        problem_summary=field_values["problem_summary"],
        root_cause=field_values["root_cause"],
        resolution=field_values["resolution"],
        diagnosis_steps=field_values["diagnosis_steps"],
        affected_modules=affected_modules,
        raw_text_hash=raw_hash,
    )

    chunks: list[CaseChunk] = []
    chunk_idx = 0

    for heading, _field_name, content in sections:
        if not content.strip():
            continue
        pos = 0
        while pos < len(content):
            chunk_text = content[pos : pos + MAX_CHUNK_CHARS].strip()
            if chunk_text:
                chunks.append(CaseChunk(
                    chunk_id=f"{case_id}#{chunk_idx}",
                    case_id=case_id,
                    source_path=source_path,
                    section=heading,
                    heading_path=f"## {heading}",
                    chunk_text=chunk_text,
                ))
                chunk_idx += 1
            pos += MAX_CHUNK_CHARS

    if not chunks and body.strip():
        chunks.append(CaseChunk(
            chunk_id=f"{case_id}#0",
            case_id=case_id,
            source_path=source_path,
            section="full",
            heading_path="",
            chunk_text=body[:MAX_CHUNK_CHARS].strip(),
        ))

    return record, chunks
