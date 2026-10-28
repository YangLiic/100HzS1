#!/bin/bash
echo "Current path: $(pwd)"

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <source_data_root> <target_data_root> <task_name1> [task_name2 ...]"
    echo "Example Single Task:   $0 ../../datasets/drobotics_raw ../../datasets/drrm_drobotics_processed task1_name"
    echo "Example Multiple Tasks: $0 ../../datasets/drobotics_raw ../../datasets/drrm_drobotics_processed task1_name task2_name task3_name"
    exit 1
fi

export SRC_ROOT=$1
export DST_ROOT=$2
shift 2

# Retrieve all remaining arguments as task list
task_list=("$@")

process_task() {
    task=$1
    echo "========= Start processing task: $task ========="

    src_dir="$SRC_ROOT/$task"
    echo "Source path: $src_dir"
    
    if [ -d "$src_dir" ]; then
        abs_path="$(realpath "$src_dir" 2>/dev/null || echo "Path resolution failed")"
        echo "Source absolute path: $abs_path"
    else
        echo "Source path does not exist, skipping task: $task"
        return 1
    fi

    target_dir="$DST_ROOT/$task"
    echo "Actual target path: $target_dir"
    
    # Call the python preprocessing script
    python tools/preprocess_drobotics.py \
        --task "$task" \
        --src_dir "$src_dir" \
        --dst_dir "$target_dir" \
        --repo "D-robotics/$task" \
        --fps 30 \
        --overwrite
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