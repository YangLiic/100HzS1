#!/usr/bin/env bash
set -euo pipefail
VFA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBOTWIN_ROOT="$VFA_ROOT/simulation/robotwin2.0"
PYTHON="${ROBOTWIN_ENV:-$ROBOTWIN_ROOT/.venv}/bin/python"
[ -x "$PYTHON" ] || { echo "Run simulation/robotwin2.0/script/setup_env.sh first"; exit 1; }
# Use only this checkout's runtime and its unified environment.
unset PYTHONHOME ROBOTWIN_VENV_SITE VFA_REMOTE
export PYTHONPATH="$VFA_ROOT/src:$ROBOTWIN_ROOT:$ROBOTWIN_ROOT/policy"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
source "$ROBOTWIN_ROOT/script/headless.sh"
cd "$ROBOTWIN_ROOT"
exec "$PYTHON" -u "$ROBOTWIN_ROOT/policy/VFA/evaluate.py" "$@"
