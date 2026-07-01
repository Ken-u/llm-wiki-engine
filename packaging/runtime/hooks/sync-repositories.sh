#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${RUNTIME_CONFIG:-runtime-config.yaml}}"

if command -v python3 >/dev/null 2>&1; then
  exec python3 "${SCRIPT_DIR}/sync-repositories.py" "${CONFIG_PATH}"
fi

if command -v python >/dev/null 2>&1; then
  exec python "${SCRIPT_DIR}/sync-repositories.py" "${CONFIG_PATH}"
fi

echo "python3 or python is required to run sync-repositories.py" >&2
exit 127
