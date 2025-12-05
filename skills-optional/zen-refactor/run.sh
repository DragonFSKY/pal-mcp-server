#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_NAME="$(basename "${SCRIPT_DIR}")"

# Find the runner script (check both installed and development locations)
if [[ -f "${SCRIPT_DIR}/scripts/zen_skill_runner.py" ]]; then
    RUNNER="${SCRIPT_DIR}/scripts/zen_skill_runner.py"
elif [[ -f "${SCRIPT_DIR}/../scripts/zen_skill_runner.py" ]]; then
    RUNNER="${SCRIPT_DIR}/../scripts/zen_skill_runner.py"
else
    echo "Error: zen_skill_runner.py not found" >&2
    exit 1
fi

# Read ZEN_MCP_ROOT from config file if exists
CONFIG_FILE="${SCRIPT_DIR}/scripts/.zen_config"
if [[ -f "$CONFIG_FILE" ]]; then
    ZEN_MCP_ROOT="$(sed -n 's/^ZEN_MCP_ROOT="\([^"]*\)".*/\1/p' "$CONFIG_FILE" 2>/dev/null || true)"
    if [[ -n "$ZEN_MCP_ROOT" ]]; then
        export ZEN_MCP_SERVER_ROOT="$ZEN_MCP_ROOT"
    fi
fi

# Determine Python executable
if [[ -n "${ZEN_MCP_SERVER_ROOT:-}" ]] && [[ -x "$ZEN_MCP_SERVER_ROOT/.zen_venv/bin/python" ]]; then
    PYTHON_BIN="$ZEN_MCP_SERVER_ROOT/.zen_venv/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

# Execute the runner with the skill name and pass all arguments
exec "$PYTHON_BIN" "$RUNNER" --skill "$SKILL_NAME" "$@"
