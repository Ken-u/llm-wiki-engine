"""Tests for failed ingest output rollback."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.ingest.pipeline import auto_ingest


def test_auto_ingest_rolls_back_written_files_when_write_fails(tmp_path):
    project = tmp_path / "project"
    source = project / "raw" / "sources" / "doc.md"
    existing = project / "wiki" / "existing.md"
    new_file = project / "wiki" / "new.md"
    source.parent.mkdir(parents=True)
    existing.parent.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")
    existing.write_text("old", encoding="utf-8")

    generation = "\n".join([
        "---FILE: wiki/existing.md---",
        "new",
        "---END FILE---",
        "---FILE: wiki/new.md---",
        "new file",
        "---END FILE---",
    ])

    async def failing_write(project_dir, blocks, source_filename):
        (Path(project_dir) / "wiki" / "existing.md").write_text("new", encoding="utf-8")
        (Path(project_dir) / "wiki" / "new.md").write_text("new file", encoding="utf-8")
        raise RuntimeError("disk write failed")

    async def run():
        with patch("app.ingest.pipeline.llm_client.stream_collect", AsyncMock(side_effect=["analysis", generation])):
            with patch("app.ingest.pipeline.write_file_blocks", failing_write):
                try:
                    await auto_ingest(str(project), str(source))
                except RuntimeError as exc:
                    assert "disk write failed" in str(exc)
                else:
                    raise AssertionError("write failure should fail the ingest")

    asyncio.run(run())

    assert existing.read_text(encoding="utf-8") == "old"
    assert not new_file.exists()
