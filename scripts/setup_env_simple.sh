#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[setup_env_simple] %s\n' "$*" >&2
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_FILE="${PROJECT_ROOT}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${ENV_FILE}"
  set +a
  log "Loaded environment variables from ${ENV_FILE}"
else
  log "No .env file found at ${ENV_FILE}; continuing without it"
fi

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv CLI"
  curl -LsSf https://astral.sh/uv/install.sh | sh
else
  log "uv already installed at $(command -v uv)"
fi

log "Simple environment bootstrap complete"
