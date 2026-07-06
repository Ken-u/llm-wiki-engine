"""Runtime bundle packaging and extraction helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from argparse import ArgumentParser
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


@dataclass(frozen=True)
class BundlePackSummary:
    output_path: str
    hash: str
    knowledge: str
    cases: str
    config: str
    hooks: str


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


EXCLUDED_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}

EXCLUDED_DIRS = {
    "__pycache__",
    ".pytest_cache",
}


def _should_exclude(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES:
        return True
    if path.name.startswith(".") and path.name.endswith(".tmp"):
        return True
    return any(part in EXCLUDED_DIRS for part in path.parts)


def _iter_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and not _should_exclude(path.relative_to(root))),
        key=lambda p: p.relative_to(root).as_posix(),
    )


def _write_tree(zf: zipfile.ZipFile, source: Path, archive_root: str) -> None:
    for file_path in _iter_files(source):
        rel = file_path.relative_to(source).as_posix()
        zf.write(file_path, f"{archive_root.rstrip('/')}/{rel}")


def _validate_pack_inputs(
    knowledge_path: Path,
    output_path: Path,
    cases_path: Path | None,
    config_path: Path | None,
    hooks_path: Path | None,
    force: bool,
) -> None:
    if not knowledge_path.is_dir():
        raise BundleError(f"Knowledge directory not found: {knowledge_path}")
    if not (knowledge_path / "wiki").is_dir():
        raise BundleError(f"Knowledge directory must contain wiki/: {knowledge_path}")
    if cases_path is not None and not cases_path.is_dir():
        raise BundleError(f"Case library directory not found: {cases_path}")
    if config_path is not None and not config_path.is_file():
        raise BundleError(f"Runtime config file not found: {config_path}")
    if hooks_path is not None and not hooks_path.is_dir():
        raise BundleError(f"Hooks directory not found: {hooks_path}")
    if output_path.exists() and not force:
        raise BundleError(f"Output bundle already exists: {output_path}. Use --force to overwrite.")


def pack_runtime_bundle(
    *,
    knowledge_path: str | Path,
    output_path: str | Path,
    cases_path: str | Path | None = None,
    config_path: str | Path | None = None,
    hooks_path: str | Path | None = None,
    force: bool = False,
) -> BundlePackSummary:
    knowledge = Path(knowledge_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    cases = Path(cases_path).expanduser().resolve() if cases_path is not None else None
    config = Path(config_path).expanduser().resolve() if config_path is not None else None
    hooks = Path(hooks_path).expanduser().resolve() if hooks_path is not None else None

    _validate_pack_inputs(knowledge, output, cases, config, hooks, force)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f".{output.name}.tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    try:
        with zipfile.ZipFile(tmp_output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if config is not None:
                zf.write(config, DEFAULT_CONFIG_NAME)
            _write_tree(zf, knowledge, "data/knowledge")
            if cases is not None:
                _write_tree(zf, cases, "data/cases")
            if hooks is not None:
                _write_tree(zf, hooks, "hooks")
        tmp_output.replace(output)
    except Exception:
        tmp_output.unlink(missing_ok=True)
        raise

    return BundlePackSummary(
        output_path=str(output),
        hash=_sha256_file(output),
        knowledge="data/knowledge",
        cases="data/cases" if cases is not None else "",
        config=DEFAULT_CONFIG_NAME if config is not None else "",
        hooks="hooks" if hooks is not None else "",
    )


def _build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Runtime bundle tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack", help="Package a runtime zip bundle")
    pack.add_argument("--knowledge", required=True)
    pack.add_argument("--cases")
    pack.add_argument("--config")
    pack.add_argument("--hooks")
    pack.add_argument("--output", required=True)
    pack.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "pack":
        summary = pack_runtime_bundle(
            knowledge_path=args.knowledge,
            output_path=args.output,
            cases_path=args.cases,
            config_path=args.config,
            hooks_path=args.hooks,
            force=args.force,
        )
        print(f"Bundle written: {summary.output_path}")
        print(f"Knowledge: {summary.knowledge}")
        print(f"Cases: {summary.cases or '-'}")
        print(f"Config: {summary.config or '-'}")
        print(f"Hooks: {summary.hooks or '-'}")
        print(f"SHA256: {summary.hash}")


if __name__ == "__main__":
    main()
