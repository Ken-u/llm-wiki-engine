from __future__ import annotations

import os
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
