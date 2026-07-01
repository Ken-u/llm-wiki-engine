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
  uv sync --extra dev
  uv run pyinstaller --clean --noconfirm packaging/runtime/llm-wiki-runtime.spec
  mkdir -p dist/runtime
  cp dist/llm-wiki-runtime dist/runtime/llm-wiki-runtime-linux-x86_64
  cp runtime-config.example.yaml dist/runtime/runtime-config.example.yaml
  echo "Linux runtime binary written to dist/runtime/llm-wiki-runtime-linux-x86_64"
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
