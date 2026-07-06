"""Runtime bundle packaging and extraction helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


MARKER_FILE = ".llmwiki-bundle-extracted"
DEFAULT_CONFIG_NAME = "runtime-config.yaml"


class BundleError(RuntimeError):
    """Raised when a runtime bundle cannot be prepared."""


@dataclass(frozen=True)
class BundleInfo:
    enabled: bool
    path: str
    hash: str
    extract_dir: str
    config_path: str

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


_bundle_info: BundleInfo | None = None


def get_runtime_bundle_info() -> BundleInfo | None:
    return _bundle_info


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_cache_dir(bundle_path: Path) -> Path:
    configured = os.environ.get("LLMWIKI_RUNTIME_BUNDLE_CACHE")
    if configured:
        return Path(configured).expanduser()

    home = Path.home()
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root) / "llm-wiki-runtime" / "bundles"
    if home.exists():
        if os.name == "posix" and os.uname().sysname == "Darwin":
            return home / "Library" / "Caches" / "llm-wiki-runtime" / "bundles"
        return home / ".cache" / "llm-wiki-runtime" / "bundles"
    return bundle_path.parent / ".runtime-bundles"


def _ensure_cache_root(cache_root: Path, bundle_path: Path) -> Path:
    try:
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root
    except OSError:
        fallback = bundle_path.parent / ".runtime-bundles"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback.resolve()


def _is_unsafe_zip_name(name: str) -> bool:
    if not name:
        return True
    normalized = name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute():
        return True
    if normalized.startswith("//") or name.startswith("\\\\"):
        return True
    return any(part in ("", "..") for part in posix.parts)


def _validate_zip_entry(name: str, destination: Path) -> Path:
    if _is_unsafe_zip_name(name):
        raise BundleError(f"Unsafe bundle entry: {name}")
    candidate = (destination / name.replace("\\", "/")).resolve()
    root = destination.resolve()
    if root != candidate and root not in candidate.parents:
        raise BundleError(f"Unsafe bundle entry: {name}")
    return candidate


def _extract_zip_safely(bundle_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(bundle_path) as zf:
        for info in zf.infolist():
            target = _validate_zip_entry(info.filename, destination)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def prepare_runtime_bundle(
    bundle_path: str | Path,
    *,
    external_config_path: str | Path | None = None,
) -> BundleInfo:
    global _bundle_info

    path = Path(bundle_path).expanduser().resolve()
    if not path.is_file():
        raise BundleError(f"Runtime bundle not found: {path}")
    if not zipfile.is_zipfile(path):
        raise BundleError(f"Runtime bundle is not a zip archive: {path}")

    bundle_hash = _sha256_file(path)
    cache_root = _default_cache_dir(path).expanduser().resolve()
    extract_dir = cache_root / bundle_hash
    marker = extract_dir / MARKER_FILE

    if not marker.is_file():
        cache_root = _ensure_cache_root(cache_root, path)
        extract_dir = cache_root / bundle_hash
        marker = extract_dir / MARKER_FILE
        tmp_dir = Path(tempfile.mkdtemp(prefix=f".{bundle_hash}.", dir=str(cache_root)))
        try:
            _extract_zip_safely(path, tmp_dir)
            if not (tmp_dir / "data" / "knowledge").is_dir():
                raise BundleError("Runtime bundle must contain data/knowledge")
            marker_tmp = tmp_dir / MARKER_FILE
            marker_tmp.write_text(bundle_hash, encoding="utf-8")
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            tmp_dir.replace(extract_dir)
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    os.environ["RUNTIME_BUNDLE_DIR"] = str(extract_dir)
    bundled_config = extract_dir / DEFAULT_CONFIG_NAME
    if bundled_config.is_file():
        config_path = bundled_config
    elif external_config_path is not None:
        config_path = Path(external_config_path).expanduser().resolve()
        if not config_path.is_file():
            raise BundleError(f"Runtime config not found: {config_path}")
    else:
        raise BundleError(
            "Runtime bundle does not contain runtime-config.yaml; "
            "add runtime-config.yaml to the bundle or start with --config."
        )

    _bundle_info = BundleInfo(
        enabled=True,
        path=str(path),
        hash=bundle_hash,
        extract_dir=str(extract_dir),
        config_path=str(config_path),
    )
    return _bundle_info
