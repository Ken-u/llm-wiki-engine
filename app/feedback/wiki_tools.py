"""Wiki filesystem tools for the Compiler Agent.

Provides Read / Glob / Edit / Grep / Create tools that operate on a
*working copy* of the wiki directory.  The original is never touched
until the user explicitly approves and applies the repair.

WorkingCopy lifecycle:
  1. ``create()``  — snapshot the live wiki into a temp dir
  2.  Agent calls tools against the working copy
  3. ``collect_changes()`` — diff the working copy vs the original
  4. ``cleanup()``  — remove the temp dir
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "读取 wiki 中指定文件的内容。路径相对于 wiki 根目录。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对于 wiki 根目录的文件路径，例如 'index.md' 或 'sources/page.md'",
                },
            },
            "required": ["path"],
        },
    },
}

TOOL_LIST_FILES = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "列出 wiki 目录中匹配 glob 模式的文件。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob 模式，例如 '*.md'、'sources/**/*.md'、'**/*'",
                },
            },
            "required": ["pattern"],
        },
    },
}

TOOL_GREP = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": "在 wiki 文件中搜索匹配正则表达式的内容行。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "正则表达式搜索模式",
                },
                "path": {
                    "type": "string",
                    "description": "限定搜索范围的目录或文件路径（相对 wiki 根）。省略则搜索全部。",
                    "default": "",
                },
                "ignore_case": {
                    "type": "boolean",
                    "description": "是否忽略大小写，默认 false",
                    "default": False,
                },
            },
            "required": ["pattern"],
        },
    },
}

TOOL_EDIT_FILE = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "编辑 wiki 文件：将文件中唯一匹配的 old_string 替换为 new_string。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对 wiki 根目录的文件路径",
                },
                "old_string": {
                    "type": "string",
                    "description": "要替换的原始文本（必须在文件中唯一匹配）",
                },
                "new_string": {
                    "type": "string",
                    "description": "替换后的文本",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
}

TOOL_CREATE_FILE = {
    "type": "function",
    "function": {
        "name": "create_file",
        "description": "在 wiki 中创建新文件。如果文件已存在则覆盖写入。",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对 wiki 根目录的文件路径",
                },
                "content": {
                    "type": "string",
                    "description": "文件内容",
                },
            },
            "required": ["path", "content"],
        },
    },
}

TOOL_SUBMIT_CHANGES = {
    "type": "function",
    "function": {
        "name": "submit_changes",
        "description": "完成所有修改后调用此工具提交变更。必须在所有编辑完成后调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "本次修复的整体变更摘要",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "对本次修复质量的置信度",
                },
            },
            "required": ["summary", "confidence"],
        },
    },
}

ALL_TOOLS = [
    TOOL_READ_FILE,
    TOOL_LIST_FILES,
    TOOL_GREP,
    TOOL_EDIT_FILE,
    TOOL_CREATE_FILE,
    TOOL_SUBMIT_CHANGES,
]


# ---------------------------------------------------------------------------
# FileChange dataclass — one per modified / created file
# ---------------------------------------------------------------------------

@dataclass
class FileChange:
    path: str
    action: str  # "modify" | "create" | "delete"
    old_content: str | None
    new_content: str


# ---------------------------------------------------------------------------
# WorkingCopy — manages a temp snapshot of the wiki
# ---------------------------------------------------------------------------

@dataclass
class WorkingCopy:
    """Temporary copy of a project wiki for safe agent editing."""
    original_wiki_dir: str
    work_dir: str = ""
    _created: bool = field(default=False, repr=False)

    def create(self) -> None:
        self.work_dir = tempfile.mkdtemp(prefix="wiki_repair_")
        if os.path.isdir(self.original_wiki_dir):
            shutil.copytree(
                self.original_wiki_dir,
                os.path.join(self.work_dir, "wiki"),
                dirs_exist_ok=True,
            )
        else:
            os.makedirs(os.path.join(self.work_dir, "wiki"), exist_ok=True)
        self._created = True
        logger.info("WorkingCopy created at %s", self.work_dir)

    @property
    def wiki_root(self) -> str:
        return os.path.join(self.work_dir, "wiki")

    def cleanup(self) -> None:
        if self._created and self.work_dir and os.path.isdir(self.work_dir):
            shutil.rmtree(self.work_dir, ignore_errors=True)
            logger.info("WorkingCopy cleaned up: %s", self.work_dir)
            self._created = False

    def _resolve(self, rel_path: str) -> str:
        """Resolve a relative path to an absolute path inside the working copy.
        Raises ValueError on path traversal attempts."""
        clean = os.path.normpath(rel_path)
        if clean.startswith("..") or os.path.isabs(clean):
            raise ValueError(f"Path traversal not allowed: {rel_path}")
        return os.path.join(self.wiki_root, clean)

    # --- tool implementations ------------------------------------------------

    def read_file(self, path: str) -> str:
        try:
            fp = self._resolve(path)
        except ValueError as e:
            return f"[ERROR] {e}"
        if not os.path.isfile(fp):
            return f"[ERROR] File not found: {path}"
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()

    def list_files(self, pattern: str) -> str:
        root = Path(self.wiki_root)
        if not root.exists():
            return "[]"
        matches: list[str] = []
        for p in root.rglob("*"):
            if p.is_file():
                rel = str(p.relative_to(root))
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(p.name, pattern):
                    matches.append(rel)
        matches.sort()
        return "\n".join(matches) if matches else "(no files matched)"

    def grep(self, pattern: str, path: str = "", ignore_case: bool = False) -> str:
        root = Path(self.wiki_root)
        search_root = root / path if path else root
        if not search_root.exists():
            return f"[ERROR] Path not found: {path}"

        flags = re.IGNORECASE if ignore_case else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"[ERROR] Invalid regex: {e}"

        results: list[str] = []
        files = [search_root] if search_root.is_file() else sorted(search_root.rglob("*"))
        for fp in files:
            if not fp.is_file():
                continue
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            rel = str(fp.relative_to(root))
                            results.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(results) >= 200:
                                results.append("... (truncated at 200 matches)")
                                return "\n".join(results)
            except Exception:
                continue
        return "\n".join(results) if results else "(no matches)"

    def edit_file(self, path: str, old_string: str, new_string: str) -> str:
        try:
            fp = self._resolve(path)
        except ValueError as e:
            return f"[ERROR] {e}"
        if not os.path.isfile(fp):
            return f"[ERROR] File not found: {path}"
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
        count = content.count(old_string)
        if count == 0:
            return f"[ERROR] old_string not found in {path}"
        if count > 1:
            return f"[ERROR] old_string matches {count} times in {path}. Must be unique."
        new_content = content.replace(old_string, new_string, 1)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"OK: edited {path}"

    def create_file(self, path: str, content: str) -> str:
        try:
            fp = self._resolve(path)
        except ValueError as e:
            return f"[ERROR] {e}"
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(content)
        return f"OK: created {path}"

    # --- diff collection -----------------------------------------------------

    def collect_changes(self) -> list[FileChange]:
        """Compare working copy against the original wiki and return changes."""
        orig_root = Path(self.original_wiki_dir)
        work_root = Path(self.wiki_root)
        changes: list[FileChange] = []

        work_files: set[str] = set()
        if work_root.exists():
            for p in work_root.rglob("*"):
                if p.is_file():
                    work_files.add(str(p.relative_to(work_root)))

        orig_files: set[str] = set()
        if orig_root.exists():
            for p in orig_root.rglob("*"):
                if p.is_file():
                    orig_files.add(str(p.relative_to(orig_root)))

        for rel in sorted(work_files):
            work_path = work_root / rel
            orig_path = orig_root / rel
            new_content = work_path.read_text(encoding="utf-8", errors="replace")

            if rel in orig_files:
                old_content = orig_path.read_text(encoding="utf-8", errors="replace")
                if old_content != new_content:
                    changes.append(FileChange(
                        path=rel, action="modify",
                        old_content=old_content, new_content=new_content,
                    ))
            else:
                changes.append(FileChange(
                    path=rel, action="create",
                    old_content=None, new_content=new_content,
                ))

        return changes


# ---------------------------------------------------------------------------
# Tool executor — dispatches a tool call to the WorkingCopy
# ---------------------------------------------------------------------------

def execute_tool(wc: WorkingCopy, name: str, args: dict) -> str:
    """Execute a tool call and return the result string."""
    try:
        if name == "read_file":
            return wc.read_file(args.get("path", ""))
        elif name == "list_files":
            return wc.list_files(args.get("pattern", "*"))
        elif name == "grep":
            return wc.grep(
                args.get("pattern", ""),
                args.get("path", ""),
                args.get("ignore_case", False),
            )
        elif name == "edit_file":
            return wc.edit_file(
                args.get("path", ""),
                args.get("old_string", ""),
                args.get("new_string", ""),
            )
        elif name == "create_file":
            return wc.create_file(
                args.get("path", ""),
                args.get("content", ""),
            )
        elif name == "submit_changes":
            return "OK: changes submitted"
        else:
            return f"[ERROR] Unknown tool: {name}"
    except ValueError as e:
        return f"[ERROR] {e}"
    except Exception as e:
        logger.exception("Tool %s execution error", name)
        return f"[ERROR] {type(e).__name__}: {e}"
