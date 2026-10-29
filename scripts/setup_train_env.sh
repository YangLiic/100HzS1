#!/usr/bin/env bash
# Usage: bash scripts/setup_train_env.sh [--refresh-lock]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
ENV_DIR="${S1_TRAIN_ENV:-$ROOT/.venv}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
[ -n "$UV_BIN" ] || UV_BIN="$HOME/.local/bin/uv"
[ -x "$UV_BIN" ] || { echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export TMPDIR="${S1_SETUP_TMPDIR:-$ROOT/.cache/tmp}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
if [ ! -f "$ENV_DIR/pyvenv.cfg" ]; then
    "$UV_BIN" venv --python "$PYTHON_BIN" "$ENV_DIR"
fi
if [ ! -f requirements_train.lock ] || [ "${1:-}" = "--refresh-lock" ]; then
    "$UV_BIN" pip compile --python-version 3.10 --torch-backend cu126 --generate-hashes \
        requirements_train.txt -o requirements_train.lock
fi
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" --torch-backend cu126 \
    --require-hashes -r requirements_train.lock
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" --no-deps -e .
"$UV_BIN" pip check --python "$ENV_DIR/bin/python"
echo "100HzS1 training environment: $ENV_DIR"
