#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/script/setup_env.sh" "$@"
bash "$ROOT/script/setup_curobo.sh"
