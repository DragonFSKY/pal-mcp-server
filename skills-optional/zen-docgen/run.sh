#!/usr/bin/env bash
set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_NAME="$(basename "${SCRIPT_DIR}")"

# Read ZEN_MCP_ROOT from config file if exists (installed mode)
CONFIG_FILE="${SCRIPT_DIR}/scripts/.zen_config"
if [[ -f "$CONFIG_FILE" ]]; then
    ZEN_MCP_ROOT="$(sed -n 's/^ZEN_MCP_ROOT="\([^"]*\)".*/\1/p' "$CONFIG_FILE" 2>/dev/null || true)"
    if [[ -n "$ZEN_MCP_ROOT" ]]; then
        export ZEN_MCP_SERVER_ROOT="$ZEN_MCP_ROOT"
    fi
fi

# Find the runner script
# Priority: 1) config-based path, 2) installed location, 3) development locations (up to 4 levels)
find_runner() {
    # 1) Use ZEN_MCP_ROOT if available (from .zen_config or environment)
    if [[ -n "${ZEN_MCP_SERVER_ROOT:-}" ]] && [[ -f "${ZEN_MCP_SERVER_ROOT}/scripts/zen_skill_runner.py" ]]; then
        echo "${ZEN_MCP_SERVER_ROOT}/scripts/zen_skill_runner.py"
        return 0
    fi

    # 2) Check installed location (runner copied to skill directory)
    if [[ -f "${SCRIPT_DIR}/scripts/zen_skill_runner.py" ]]; then
        echo "${SCRIPT_DIR}/scripts/zen_skill_runner.py"
        return 0
    fi

    # 3) Search upward for development locations (skills/, skills-optional/, etc.)
    local current="${SCRIPT_DIR}"
    for _ in {1..4}; do
        current="$(dirname "$current")"
        if [[ -f "$current/scripts/zen_skill_runner.py" ]]; then
            echo "$current/scripts/zen_skill_runner.py"
            return 0
        fi
    done

    return 1
}

RUNNER="$(find_runner)" || {
    echo "Error: zen_skill_runner.py not found" >&2
    echo "Searched: config path, ${SCRIPT_DIR}/scripts/, and up to 4 parent directories" >&2
    echo "Tip: Run from zen-mcp-server repo or execute ./install-skills.sh first" >&2
    exit 1
}

# Determine Python executable
if [[ -n "${ZEN_MCP_SERVER_ROOT:-}" ]] && [[ -x "$ZEN_MCP_SERVER_ROOT/.zen_venv/bin/python" ]]; then
    PYTHON_BIN="$ZEN_MCP_SERVER_ROOT/.zen_venv/bin/python"
else
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

# Execute the runner with the skill name and pass all arguments
exec "$PYTHON_BIN" "$RUNNER" --skill "$SKILL_NAME" "$@"
