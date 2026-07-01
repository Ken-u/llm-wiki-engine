#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${RUNTIME_DIR}/runtime-config.yaml}"

exec "${RUNTIME_DIR}/llm-wiki-runtime" --config "${CONFIG_PATH}" --sync-repositories
