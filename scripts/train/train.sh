#!/usr/bin/env bash
# Usage: train.sh [config.yaml] [Hydra overrides...] [--dry-run]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
    cat <<'HELP'
Usage: scripts/train/train.sh [config.yaml] [Hydra overrides...] [--dry-run]
Default config: configs/vfa_train/dit_fm_dinov3cnnl_1v.yaml
Default checkpoints: /DRoboticsAILab/users/licheng.yang/checkpoints/VFA

Environment:
  CONFIG_PATH         Used when no config file argument is given.
  PYTHON_BIN          Defaults to the repository .venv/bin/python.
  GPU_IDS             GPU list, e.g. 0,1; defaults to CUDA_VISIBLE_DEVICES or all.
  NUM_PROCESSES       Defaults to the selected/visible GPU count (minimum 1).
  ACCELERATE_CONFIG   Defaults to configs/vfa_accelerate_config.yaml.
  MAIN_PROCESS_PORT   Defaults to 0 for automatic port selection.
  CHECKPOINT_ROOT     Optional override of the configured checkpoint root.

Examples:
  NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 scripts/train/train.sh \
    configs/vfa_train/dit_fm_dinov3cnnl_1v.yaml \
    train_dataset.path=/absolute/path/to/dataset_parent \
    train_dataset.task=adjust_bottle train_dataset.demo=100
  scripts/train/train.sh configs/vfa_train/dit_fm_dinov3cnnl_1v.yaml \
    train_dataset.cache_to_ram=false --dry-run

The LeRobot shared cache is managed by the VFA dataset itself.
--dry-run prints the command and exits without starting training.
HELP
}
CONFIG_PATH="${CONFIG_PATH:-configs/vfa_train/dit_fm_dinov3cnnl_1v.yaml}"
dry_run=false
config_given=false
hydra_overrides=()
for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true ;;
        *)
            if ! "$config_given" && [[ "$arg" != *=* ]] && [[ "$arg" == *.yaml || "$arg" == *.yml ]]; then
                CONFIG_PATH="$arg"
                config_given=true
            else
                hydra_overrides+=("$arg")
            fi
            ;;
    esac
done
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python is not executable: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config does not exist: $CONFIG_PATH" >&2
    exit 1
fi
config_dir="$(cd "$(dirname "$CONFIG_PATH")" && pwd)"
config_name="$(basename "$CONFIG_PATH")"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${REPO_ROOT}/configs/vfa_accelerate_config.yaml}"
if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
    echo "Accelerate config does not exist: $ACCELERATE_CONFIG" >&2
    exit 1
fi
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-all}}"
if [[ -z "${NUM_PROCESSES:-}" ]]; then
    if [[ "$GPU_IDS" == all ]]; then
        NUM_PROCESSES="$("$PYTHON_BIN" -c 'import torch; print(max(1, torch.cuda.device_count()))')"
    else
        IFS=',' read -r -a selected_gpus <<< "$GPU_IDS"
        NUM_PROCESSES="${#selected_gpus[@]}"
    fi
fi
if [[ ! "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROCESSES must be a positive integer: $NUM_PROCESSES" >&2
    exit 1
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
# Environment defaults come before CLI overrides, so an explicit CLI value wins.
defaults=()
if [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
    defaults+=("checkpoint_root=${CHECKPOINT_ROOT}")
fi
CMD=(
    "$PYTHON_BIN" -m accelerate.commands.accelerate_cli launch
    --config_file "$ACCELERATE_CONFIG"
    --num_processes "$NUM_PROCESSES"
    --num_machines 1
    --gpu_ids "$GPU_IDS"
)
if (( NUM_PROCESSES > 1 )); then
    CMD+=(--multi_gpu --main_process_port "${MAIN_PROCESS_PORT:-0}")
fi
CMD+=(
    "${REPO_ROOT}/main.py"
    "--config-path=$config_dir"
    "--config-name=$config_name"
    "${defaults[@]}"
    "${hydra_overrides[@]}"
)
echo "Config: ${config_dir}/${config_name} | GPUs: ${GPU_IDS} | processes: ${NUM_PROCESSES}"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
if "$dry_run"; then
    exit 0
fi
exec "${CMD[@]}"
