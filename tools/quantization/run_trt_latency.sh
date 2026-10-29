#!/usr/bin/env bash
set -euo pipefail
cd /data/vepfs/licheng.yang/100HzS1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH=work/trt_latency_20261029/deps
task_root=work/trt_latency_20261029/artifacts
trap 'task_rc=$?; echo "RUNNER_EXIT=${task_rc}"; echo "${task_rc}" > "${task_root}/runner.exit"' EXIT
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; do
    sleep 3
done
for task_model in dino b2; do
    for task_batch in 1 16; do
        for task_precision in fp16 int8; do
            echo "STAGE ${task_model} BS${task_batch} ${task_precision}"
            .venv/bin/python tools/quantization/benchmark_trt_latency.py --root "$task_root" --mode build --model "$task_model" --batch "$task_batch" --precision "$task_precision"
            .venv/bin/python tools/quantization/benchmark_trt_latency.py --root "$task_root" --mode bench --model "$task_model" --batch "$task_batch" --precision "$task_precision"
        done
    done
done
.venv/bin/python tools/quantization/summarize_trt_latency.py --root "$task_root" --report docs/100HzS1/tensorrt_latency_20261029.md
echo ALL_BENCHMARKS_FINISHED
