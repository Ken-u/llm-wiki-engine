#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv sync --extra dev
uv run pyinstaller --clean --noconfirm packaging/runtime/llm-wiki-runtime.spec

echo "Runtime binary written to dist/llm-wiki-runtime"

