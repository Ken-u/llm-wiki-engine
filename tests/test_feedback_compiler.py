"""Unit tests for compiler prompt building and wiki tools."""

import os
import tempfile

from app.feedback.compiler import (
    _build_compiler_prompt,
    CompilerInput,
)
from app.feedback.wiki_tools import (
    ALL_TOOLS,
    TOOL_SUBMIT_CHANGES,
    WorkingCopy,
    execute_tool,
)


class TestAllTools:
    def test_tool_count(self):
        assert len(ALL_TOOLS) == 6

    def test_submit_changes_tool(self):
        assert TOOL_SUBMIT_CHANGES["type"] == "function"
        assert TOOL_SUBMIT_CHANGES["function"]["name"] == "submit_changes"

    def test_all_tools_have_names(self):
        names = {t["function"]["name"] for t in ALL_TOOLS}
        assert names == {"read_file", "list_files", "grep", "edit_file", "create_file", "submit_changes"}

    def test_submit_changes_required_params(self):
        params = TOOL_SUBMIT_CHANGES["function"]["parameters"]
        assert set(params["required"]) == {"summary", "confidence"}


class TestBuildCompilerPrompt:
    def test_returns_two_messages(self):
        inp = _make_input()
        msgs = _build_compiler_prompt(inp)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_includes_evaluator_reason(self):
        inp = _make_input()
        msgs = _build_compiler_prompt(inp)
        assert "test" in msgs[1]["content"]

    def test_system_prompt_mentions_tools(self):
        inp = _make_input()
        msgs = _build_compiler_prompt(inp)
        sys_content = msgs[0]["content"]
        assert "read_file" in sys_content
        assert "edit_file" in sys_content
        assert "submit_changes" in sys_content

    def test_system_prompt_has_structure_rules(self):
        inp = _make_input()
        msgs = _build_compiler_prompt(inp)
        sys_content = msgs[0]["content"]
        assert "index.md" in sys_content
        assert "sources/" in sys_content
        assert "不要在此文件中写入正文内容" in sys_content

    def test_review_guidance_included(self):
        inp = _make_input(review_guidance="请修正第三段")
        msgs = _build_compiler_prompt(inp)
        assert "请修正第三段" in msgs[1]["content"]


class TestWorkingCopy:
    def test_create_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            wiki = os.path.join(d, "wiki")
            os.makedirs(wiki)
            with open(os.path.join(wiki, "test.md"), "w") as f:
                f.write("# Hello")

            wc = WorkingCopy(original_wiki_dir=wiki)
            wc.create()
            assert os.path.isdir(wc.wiki_root)
            assert os.path.isfile(os.path.join(wc.wiki_root, "test.md"))

            wc.cleanup()
            assert not os.path.isdir(wc.work_dir)

    def test_read_file(self):
        with tempfile.TemporaryDirectory() as d:
            wiki = os.path.join(d, "wiki")
            os.makedirs(wiki)
            with open(os.path.join(wiki, "a.md"), "w") as f:
                f.write("content A")

            wc = WorkingCopy(original_wiki_dir=wiki)
            wc.create()
            try:
                assert wc.read_file("a.md") == "content A"
                assert "[ERROR]" in wc.read_file("missing.md")
            finally:
                wc.cleanup()

    def test_edit_file(self):
        with tempfile.TemporaryDirectory() as d:
            wiki = os.path.join(d, "wiki")
            os.makedirs(wiki)
            with open(os.path.join(wiki, "a.md"), "w") as f:
                f.write("hello world")

            wc = WorkingCopy(original_wiki_dir=wiki)
            wc.create()
            try:
                result = wc.edit_file("a.md", "hello", "goodbye")
                assert "OK" in result
                assert wc.read_file("a.md") == "goodbye world"
            finally:
                wc.cleanup()

    def test_edit_file_unique_check(self):
        with tempfile.TemporaryDirectory() as d:
            wiki = os.path.join(d, "wiki")
            os.makedirs(wiki)
            with open(os.path.join(wiki, "a.md"), "w") as f:
                f.write("aaa aaa aaa")

            wc = WorkingCopy(original_wiki_dir=wiki)
            wc.create()
            try:
                result = wc.edit_file("a.md", "aaa", "bbb")
                assert "[ERROR]" in result
                assert "3 times" in result
            finally:
                wc.cleanup()

    def test_create_file(self):
        with tempfile.TemporaryDirectory() as d:
            wiki = os.path.join(d, "wiki")
            os.makedirs(wiki)

            wc = WorkingCopy(original_wiki_dir=wiki)
            wc.create()
            try:
                result = wc.create_file("new/page.md", "# New Page")
                assert "OK" in result
                assert wc.read_file("new/page.md") == "# New Page"
            finally:
                wc.cleanup()

    def test_list_files(self):
        with tempfile.TemporaryDirectory() as d:
            wiki = os.path.join(d, "wiki")
            os.makedirs(os.path.join(wiki, "sub"))
            with open(os.path.join(wiki, "a.md"), "w") as f:
                f.write("a")
            with open(os.path.join(wiki, "sub", "b.md"), "w") as f:
                f.write("b")

            wc = WorkingCopy(original_wiki_dir=wiki)
            wc.create()
            try:
                result = wc.list_files("*.md")
                assert "a.md" in result
                assert "b.md" in result
            finally:
                wc.cleanup()

    def test_grep(self):
        with tempfile.TemporaryDirectory() as d:
            wiki = os.path.join(d, "wiki")
            os.makedirs(wiki)
            with open(os.path.join(wiki, "a.md"), "w") as f:
                f.write("line1 match\nline2\nline3 match")

            wc = WorkingCopy(original_wiki_dir=wiki)
            wc.create()
            try:
                result = wc.grep("match")
                assert "a.md:1:" in result
                assert "a.md:3:" in result
            finally:
                wc.cleanup()

    def test_collect_changes(self):
        with tempfile.TemporaryDirectory() as d:
            wiki = os.path.join(d, "wiki")
            os.makedirs(wiki)
            with open(os.path.join(wiki, "existing.md"), "w") as f:
                f.write("old content")

            wc = WorkingCopy(original_wiki_dir=wiki)
            wc.create()
            try:
                wc.edit_file("existing.md", "old", "new")
                wc.create_file("brand_new.md", "# New")
                changes = wc.collect_changes()
                assert len(changes) == 2

                by_action = {c.action: c for c in changes}
                assert "modify" in by_action
                assert by_action["modify"].path == "existing.md"
                assert by_action["modify"].old_content == "old content"
                assert by_action["modify"].new_content == "new content"
                assert "create" in by_action
                assert by_action["create"].path == "brand_new.md"
            finally:
                wc.cleanup()

    def test_path_traversal_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            wiki = os.path.join(d, "wiki")
            os.makedirs(wiki)

            wc = WorkingCopy(original_wiki_dir=wiki)
            wc.create()
            try:
                result = wc.read_file("../../etc/passwd")
                assert "[ERROR]" in result
            finally:
                wc.cleanup()


class TestExecuteTool:
    def test_unknown_tool(self):
        with tempfile.TemporaryDirectory() as d:
            wiki = os.path.join(d, "wiki")
            os.makedirs(wiki)
            wc = WorkingCopy(original_wiki_dir=wiki)
            wc.create()
            try:
                result = execute_tool(wc, "nonexistent", {})
                assert "[ERROR]" in result
            finally:
                wc.cleanup()


def _make_input(**kwargs) -> CompilerInput:
    defaults = {
        "user_message": "test",
        "assistant_answer": "answer",
        "evaluator_result": {"needs_repair": True, "confidence": "high", "reason": "test"},
        "review_guidance": None,
        "raw_reads": [],
        "wiki_dir": "/tmp/test-wiki",
    }
    defaults.update(kwargs)
    return CompilerInput(**defaults)
