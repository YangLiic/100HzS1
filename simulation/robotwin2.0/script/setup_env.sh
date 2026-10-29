#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.10}"
ENV_DIR="${ROBOTWIN_ENV:-$ROOT/.venv}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
[ -n "$UV_BIN" ] || UV_BIN="$HOME/.local/bin/uv"
[ -x "$UV_BIN" ] || { echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export TMPDIR="${S1_SETUP_TMPDIR:-$ROOT/.cache/tmp}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
if [ ! -f "$ENV_DIR/pyvenv.cfg" ]; then
  "$UV_BIN" venv --python "$PYTHON_BIN" "$ENV_DIR"
fi
if [ ! -f "$ROOT/script/requirements.lock" ] || [ "${1:-}" = "--refresh-lock" ]; then
  "$UV_BIN" pip compile --python-version 3.10 --torch-backend cu126 --generate-hashes \
    "$ROOT/script/requirements.txt" -o "$ROOT/script/requirements.lock"
fi
"$UV_BIN" pip install --python "$ENV_DIR/bin/python" --torch-backend cu126 \
  --require-hashes -r "$ROOT/script/requirements.lock"
if [ "${S1_INSTALL_TENSORRT:-0}" = "1" ]; then
  "$UV_BIN" pip install --python "$ENV_DIR/bin/python" \
    -r "$ROOT/script/requirements-tensorrt.txt"
fi
"$UV_BIN" pip check --python "$ENV_DIR/bin/python"
echo "Unified runtime installed: $ENV_DIR"
