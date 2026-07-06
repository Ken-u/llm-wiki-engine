#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage:
  scripts/build-runtime.sh [--platform linux|windows|macos|all] [--ci]

Build Runtime packages from Linux.

Platforms:
  linux    Build the Linux single-file binary locally with PyInstaller.
  windows  Print the CI/Docker requirement for Windows builds.
  macos    Print the CI requirement for macOS builds.
  all      Build Linux locally and prepare instructions for Windows/macOS.

Notes:
  PyInstaller does not cross-compile Windows/macOS binaries from Linux.
  Use the GitHub Actions workflow at .github/workflows/runtime-build.yml
  to produce all three platform artifacts from one Linux-side trigger.
EOF
}

PLATFORM="linux"
CI_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)
      PLATFORM="${2:-}"
      shift 2
      ;;
    --ci)
      CI_MODE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

build_linux() {
  if [[ -d ../llm-wiki-ui && "${SKIP_RUNTIME_UI_BUILD:-0}" != "1" ]]; then
    npm --prefix ../llm-wiki-ui run build:runtime
  fi
  uv sync --extra dev
  local platform_dir="linux-x86_64"
  local outdir="dist/runtime/${platform_dir}"
  local tmpdist="dist/.tmp-runtime-build"
  local zip_path="dist/runtime-${platform_dir}.zip"
  rm -rf "$tmpdist" "$outdir"
  rm -f "$zip_path"
  uv run pyinstaller --clean --noconfirm --distpath "$tmpdist" packaging/runtime/llm-wiki-runtime.spec
  mkdir -p "$outdir"
  cp "$tmpdist/llm-wiki-runtime" "$outdir/llm-wiki-runtime"
  cp runtime-config.example.yaml "$outdir/runtime-config.example.yaml"
  cp -R packaging/runtime/hooks "$outdir/hooks"
  cp scripts/build-runtime-bundle.sh "$outdir/build-runtime-bundle.sh"
  cp scripts/build-runtime-bundle.bat "$outdir/build-runtime-bundle.bat"
  python3 - <<PY
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

platform_dir = "${platform_dir}"
outdir = Path("${outdir}")
zip_path = Path("${zip_path}")
with ZipFile(zip_path, "w", ZIP_DEFLATED) as zf:
    for path in sorted(outdir.rglob("*")):
        if path.is_file():
            zf.write(path, Path(platform_dir) / path.relative_to(outdir))
PY
  rm -rf "$tmpdist"
  echo "Linux runtime binary written to $outdir/llm-wiki-runtime"
  echo "Linux runtime package written to $zip_path"
}

unsupported_cross_build() {
  local target="$1"
  cat >&2 <<EOF
Cannot build ${target} runtime directly on Linux with PyInstaller.

Use CI to build all platforms:
  gh workflow run runtime-build.yml

Or run the native build script on ${target}.
EOF
  if [[ "$CI_MODE" -eq 1 ]]; then
    return 1
  fi
  return 0
}

case "$PLATFORM" in
  linux)
    build_linux
    ;;
  windows)
    unsupported_cross_build "Windows"
    ;;
  macos|darwin)
    unsupported_cross_build "macOS"
    ;;
  all)
    build_linux
    unsupported_cross_build "Windows"
    unsupported_cross_build "macOS"
    echo "For all platform artifacts, run GitHub Actions workflow runtime-build.yml."
    ;;
  *)
    echo "Unsupported platform: $PLATFORM" >&2
    usage >&2
    exit 2
    ;;
esac
