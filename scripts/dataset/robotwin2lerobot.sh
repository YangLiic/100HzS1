#!/bin/bash
echo "Current path: $(pwd)"

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <source_data_root> <target_data_root> <task_name1> [task_name2 ...]"
    echo "Example Single Task:   $0 ./datasets/robotwin2.0_D435_Agilex_200_rgb ./datasets/drrm_robotwin2.0_D435_Agilex_200_rgb adjust_bottle"
    echo "Example Multiple Tasks: $0 ./datasets/robotwin2.0_D435_Agilex_200_rgb ./datasets/drrm_robotwin2.0_D435_Agilex_200_rgb adjust_bottle click_bell"
    exit 1
fi

export SRC_ROOT=$1
export DST_ROOT=$2
shift 2

task_list=("$@")

process_task() {
    task=$1
    echo "========= Start processing task: $task ========="

    task_pkl=${task}/agilex_config/data
    pkl_path="$SRC_ROOT/$task_pkl"
    echo $pkl_path
    if [ -d "$pkl_path" ]; then
        abs_path="$(realpath "$pkl_path" 2>/dev/null || echo "Conversion failed")"
        echo "Absolute path: $abs_path"
    else
        echo "Path does not exist"
        return 1
    fi

    task_lerobot="${task}"
    lerobot_path="$DST_ROOT/$task_lerobot"
    echo $lerobot_path
    python tools/preprocess_robotwin2.py \
        --task $task\
        --src_dir "$pkl_path" \
        --dst_dir "$lerobot_path" \
        --repo D-robotics/$task_lerobot \
        --fps 30
}

export -f process_task

if [ ${#task_list[@]} -eq 1 ]; then
    # Single task processing without parallel
    process_task "${task_list[0]}"
else
    # Multi-task processing with parallel
    # Use GNU Parallel for parallel processing, running up to 8 processes simultaneously
    echo "${task_list[@]}" | tr ' ' '\n' | parallel -j 8 process_task {}
fi