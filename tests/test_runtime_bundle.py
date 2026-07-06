from __future__ import annotations

import os
import asyncio
import time
import zipfile
from pathlib import Path

import pytest

from app.runtime.config import load_runtime_config


def _write_zip(path: Path, files: dict[str, str | bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            if isinstance(content, str):
                zf.writestr(name, content)
            else:
                zf.writestr(name, content)


def _bundle_files(*, include_config: bool = True) -> dict[str, str]:
    files = {
        "data/knowledge/wiki/index.md": "# Bundled Knowledge\n",
    }
    if include_config:
        files["runtime-config.yaml"] = """
server:
  open_browser: false
knowledge:
  name: Bundled Knowledge
  path: ./data/knowledge
case_library:
  enabled: false
"""
    return files


def test_prepare_runtime_bundle_extracts_bundled_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.runtime.bundle import prepare_runtime_bundle

    cache = tmp_path / "cache"
    monkeypatch.setenv("LLMWIKI_RUNTIME_BUNDLE_CACHE", str(cache))
    bundle = tmp_path / "customer.llmwiki-bundle"
    _write_zip(bundle, _bundle_files())

    info = prepare_runtime_bundle(bundle)

    assert info.enabled is True
    assert info.path == str(bundle.resolve())
    assert len(info.hash) == 64
    assert info.extract_dir.startswith(str(cache.resolve()))
    assert info.config_path == str(Path(info.extract_dir, "runtime-config.yaml"))
    assert Path(info.config_path).is_file()
    assert os.environ["RUNTIME_BUNDLE_DIR"] == info.extract_dir

    settings = load_runtime_config(info.config_path, create=False)
    assert settings.knowledge.path == str(Path(info.extract_dir, "data", "knowledge").resolve())


def test_prepare_runtime_bundle_uses_external_config_when_bundle_config_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from app.runtime.bundle import prepare_runtime_bundle

    monkeypatch.setenv("LLMWIKI_RUNTIME_BUNDLE_CACHE", str(tmp_path / "cache"))
    bundle = tmp_path / "data-only.llmwiki-bundle"
    _write_zip(bundle, _bundle_files(include_config=False))
    external_config = tmp_path / "runtime-config.yaml"
    external_config.write_text(
        """
server:
  open_browser: false
knowledge:
  name: External Config
  path: ${RUNTIME_BUNDLE_DIR}/data/knowledge
case_library:
  enabled: false
""",
        encoding="utf-8",
    )

    info = prepare_runtime_bundle(bundle, external_config_path=external_config)

    assert info.config_path == str(external_config.resolve())
    assert os.environ["RUNTIME_BUNDLE_DIR"] == info.extract_dir
    settings = load_runtime_config(info.config_path, create=False)
    assert settings.knowledge.name == "External Config"
    assert settings.knowledge.path == str(Path(info.extract_dir, "data", "knowledge").resolve())


def test_prepare_runtime_bundle_requires_config_when_bundle_config_missing(tmp_path: Path):
    from app.runtime.bundle import BundleError, prepare_runtime_bundle

    bundle = tmp_path / "data-only.llmwiki-bundle"
    _write_zip(bundle, _bundle_files(include_config=False))

    with pytest.raises(BundleError, match="runtime-config.yaml"):
        prepare_runtime_bundle(bundle)


def test_prepare_runtime_bundle_reuses_extracted_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.runtime.bundle import prepare_runtime_bundle

    monkeypatch.setenv("LLMWIKI_RUNTIME_BUNDLE_CACHE", str(tmp_path / "cache"))
    bundle = tmp_path / "customer.llmwiki-bundle"
    _write_zip(bundle, _bundle_files())

    first = prepare_runtime_bundle(bundle)
    marker = Path(first.extract_dir) / ".llmwiki-bundle-extracted"
    first_mtime = marker.stat().st_mtime_ns
    time.sleep(0.001)
    second = prepare_runtime_bundle(bundle)

    assert second.extract_dir == first.extract_dir
    assert marker.stat().st_mtime_ns == first_mtime


@pytest.mark.parametrize(
    "entry_name",
    [
        "../evil.txt",
        "/tmp/evil.txt",
        "C:/tmp/evil.txt",
        r"C:\tmp\evil.txt",
        r"\\server\share\evil.txt",
    ],
)
def test_prepare_runtime_bundle_rejects_unsafe_zip_entries(tmp_path: Path, entry_name: str):
    from app.runtime.bundle import BundleError, prepare_runtime_bundle

    bundle = tmp_path / "unsafe.llmwiki-bundle"
    files = _bundle_files()
    files[entry_name] = "evil"
    _write_zip(bundle, files)

    with pytest.raises(BundleError, match="Unsafe bundle entry"):
        prepare_runtime_bundle(bundle)


def test_pack_runtime_bundle_writes_expected_layout(tmp_path: Path):
    from app.runtime.bundle import pack_runtime_bundle

    knowledge = tmp_path / "knowledge"
    (knowledge / "wiki").mkdir(parents=True)
    (knowledge / "wiki" / "index.md").write_text("# Knowledge\n", encoding="utf-8")
    (knowledge / "raw" / "sources").mkdir(parents=True)
    (knowledge / "__pycache__").mkdir()
    (knowledge / "__pycache__" / "ignored.pyc").write_bytes(b"ignored")
    cases = tmp_path / "cases"
    (cases / ".llm-wiki" / "case-index").mkdir(parents=True)
    (cases / ".llm-wiki" / "case-index" / "manifest.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "runtime-config.yaml"
    config.write_text("knowledge:\n  path: ./data/knowledge\n", encoding="utf-8")
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "startup.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    output = tmp_path / "dist" / "customer.llmwiki-bundle"

    summary = pack_runtime_bundle(
        knowledge_path=knowledge,
        output_path=output,
        cases_path=cases,
        config_path=config,
        hooks_path=hooks,
    )

    assert summary.output_path == str(output.resolve())
    assert len(summary.hash) == 64
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
    assert "runtime-config.yaml" in names
    assert "data/knowledge/wiki/index.md" in names
    assert "data/cases/.llm-wiki/case-index/manifest.json" in names
    assert "hooks/startup.sh" in names
    assert "data/knowledge/__pycache__/ignored.pyc" not in names


def test_pack_runtime_bundle_refuses_existing_output_without_force(tmp_path: Path):
    from app.runtime.bundle import BundleError, pack_runtime_bundle

    knowledge = tmp_path / "knowledge"
    (knowledge / "wiki").mkdir(parents=True)
    (knowledge / "wiki" / "index.md").write_text("# Knowledge\n", encoding="utf-8")
    output = tmp_path / "customer.llmwiki-bundle"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(BundleError, match="already exists"):
        pack_runtime_bundle(knowledge_path=knowledge, output_path=output)


def test_pack_runtime_bundle_overwrites_existing_output_with_force(tmp_path: Path):
    from app.runtime.bundle import pack_runtime_bundle

    knowledge = tmp_path / "knowledge"
    (knowledge / "wiki").mkdir(parents=True)
    (knowledge / "wiki" / "index.md").write_text("# Knowledge\n", encoding="utf-8")
    output = tmp_path / "customer.llmwiki-bundle"
    output.write_text("existing", encoding="utf-8")

    pack_runtime_bundle(knowledge_path=knowledge, output_path=output, force=True)

    assert zipfile.is_zipfile(output)


def test_resolve_runtime_config_uses_bundled_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMWIKI_RUNTIME_BUNDLE_CACHE", str(tmp_path / "cache"))
    bundle = tmp_path / "customer.llmwiki-bundle"
    _write_zip(bundle, _bundle_files())

    from app.runtime_main import _resolve_runtime_config_argument

    config_path = _resolve_runtime_config_argument(config="ignored.yaml", bundle=str(bundle))

    assert config_path.endswith("runtime-config.yaml")
    assert Path(config_path).is_file()
    assert os.environ["RUNTIME_CONFIG"] == config_path


def test_resolve_runtime_config_uses_external_config_for_data_only_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LLMWIKI_RUNTIME_BUNDLE_CACHE", str(tmp_path / "cache"))
    bundle = tmp_path / "customer.llmwiki-bundle"
    _write_zip(bundle, _bundle_files(include_config=False))
    external_config = tmp_path / "runtime-config.yaml"
    external_config.write_text(
        """
server:
  open_browser: false
knowledge:
  path: ${RUNTIME_BUNDLE_DIR}/data/knowledge
case_library:
  enabled: false
""",
        encoding="utf-8",
    )

    from app.runtime_main import _resolve_runtime_config_argument

    config_path = _resolve_runtime_config_argument(config=str(external_config), bundle=str(bundle))

    assert config_path == str(external_config.resolve())
    settings = load_runtime_config(config_path, create=False)
    assert settings.knowledge.path == str(Path(os.environ["RUNTIME_BUNDLE_DIR"], "data", "knowledge").resolve())


def test_runtime_status_includes_bundle_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LLMWIKI_RUNTIME_BUNDLE_CACHE", str(tmp_path / "cache"))
    bundle = tmp_path / "customer.llmwiki-bundle"
    _write_zip(bundle, _bundle_files())

    from app.runtime.bundle import prepare_runtime_bundle
    from app.runtime import status as runtime_status

    info = prepare_runtime_bundle(bundle)
    settings = load_runtime_config(info.config_path, create=False)
    monkeypatch.setattr(runtime_status, "_lancedb_path", lambda _project_dir: str(tmp_path / "missing-lancedb"))

    status = asyncio.run(runtime_status.build_status(settings))

    assert status["bundle"]["enabled"] is True
    assert status["bundle"]["path"] == info.path
    assert status["bundle"]["hash"] == info.hash
    assert status["bundle"]["extract_dir"] == info.extract_dir
