#!/usr/bin/env python3
"""Bench/accuracy for TRT engines on Jetson — uses torch for CUDA memory.

Usage:
  python3 bench_acc.py bench <engine> <inputs_dir> [num_runs]
  python3 bench_acc.py accuracy <ref_engine> <test_engine> <calib_dir> [num_samples]
"""
import json
import os
import sys
import time

import numpy as np
import tensorrt as trt
import torch


def load_inputs(inputs_dir):
    inputs = {}
    for f in os.listdir(inputs_dir):
        if f.endswith(".npy"):
            inputs[f[:-4]] = np.load(os.path.join(inputs_dir, f))
    return inputs


class Runner:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.input_names, self.output_names = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def run(self, sample_inputs, sync=True):
        dev = {}
        for name in self.input_names:
            arr = np.ascontiguousarray(sample_inputs[name], dtype=np.float32)
            dev[name] = torch.from_numpy(arr).cuda()
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dev[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        for name, t in dev.items():
            self.context.set_tensor_address(name, t.data_ptr())
        self.context.execute_async_v3(self.stream.cuda_stream)
        if sync:
            self.stream.synchronize()
        return {name: t.cpu().numpy() for name, t in dev.items() if name in self.output_names}


def bench(engine_path, inputs_dir, num_runs=100, warmup=10):
    runner = Runner(engine_path)
    inputs = load_inputs(inputs_dir)
    with torch.no_grad():
        for _ in range(warmup):
            runner.run(inputs)
        torch.cuda.synchronize()
        times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            runner.run(inputs)
            times.append((time.perf_counter() - t0) * 1000)
    print(f"BENCH {os.path.basename(engine_path)} runs={num_runs} "
          f"mean={np.mean(times):.2f}ms min={np.min(times):.2f}ms max={np.max(times):.2f}ms")


def accuracy(ref_engine, test_engine, calib_dir, num_samples=10):
    ref, test = Runner(ref_engine), Runner(test_engine)
    sample_dirs = sorted(
        d for d in os.listdir(calib_dir)
        if d.startswith("sample_") and os.path.isdir(os.path.join(calib_dir, d, "inputs"))
    )[:num_samples]
    metrics = {}
    with torch.no_grad():
        for sd in sample_dirs:
            inputs = load_inputs(os.path.join(calib_dir, sd, "inputs"))
            out_ref = ref.run(inputs)
            out_test = test.run(inputs)
            for name in ref.output_names:
                a, b = out_ref[name], out_test[name]
                m = metrics.setdefault(name, {"max_abs_err": [], "mean_abs_err": [], "ref_std": []})
                m["max_abs_err"].append(float(np.abs(a - b).max()))
                m["mean_abs_err"].append(float(np.abs(a - b).mean()))
                m["ref_std"].append(float(a.std()))
    print(json.dumps(
        {name: {k: float(np.mean(v)) for k, v in m.items()} for name, m in metrics.items()},
        indent=2))


def main():
    cmd = sys.argv[1]
    if cmd == "bench":
        bench(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 100)
    elif cmd == "accuracy":
        accuracy(sys.argv[2], sys.argv[3], sys.argv[4],
                 int(sys.argv[5]) if len(sys.argv) > 5 else 10)
    else:
        raise SystemExit(f"unknown cmd {cmd}")


if __name__ == "__main__":
    main()
