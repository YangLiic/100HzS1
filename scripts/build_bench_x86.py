#!/usr/bin/env python3
"""Build & bench TRT engines on x86 (L4/4090, SM 8.9) with TRT 11.

TRT 11 removed implicit-precision builder flags: FP16 must be a dtype in the
ONNX graph. So: fp32.onnx → plain build; fp16.onnx (graph-cast) → build;
int8.onnx (Q/DQ) → strongly-typed build.

Usage:
  python3 build_bench_x86.py build <onnx> <engine> [strongly_typed]
  python3 build_bench_x86.py bench <engine> <inputs_dir> [num_runs]
"""
import json
import os
import sys
import time

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart


def cu(result):
    err = result[0] if isinstance(result, tuple) else result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


def build_engine(onnx_path, engine_path, strongly_typed=False):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"parse failed: {onnx_path}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    print(f"Building {'strongly-typed' if strongly_typed else 'implicit'} engine: {onnx_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))
    print(f"ENGINE_SAVED {engine_path} {os.path.getsize(engine_path)}")


class Runner:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        cu(cudart.cudaStreamCreate())
        err, self.stream = cudart.cudaStreamCreate()
        cu(err)
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
            err, ptr = cudart.cudaMalloc(arr.nbytes)
            cu(err)
            cu(cudart.cudaMemcpy(ptr, arr, arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
            dev[name] = (ptr, arr.nbytes)
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            nbytes = int(np.prod(shape)) * 4
            err, ptr = cudart.cudaMalloc(nbytes)
            cu(err)
            dev[name] = (ptr, nbytes)
        for name, (ptr, _) in dev.items():
            self.context.set_tensor_address(name, ptr)
        self.context.execute_async_v3(self.stream)
        if sync:
            cu(cudart.cudaStreamSynchronize(self.stream))
        out = {}
        for name in self.output_names:
            ptr, nbytes = dev[name]
            arr = np.empty((nbytes // 4,), dtype=np.float32)
            cu(cudart.cudaMemcpy(arr, ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
            out[name] = arr.reshape(tuple(self.context.get_tensor_shape(name)))
        for ptr, _ in dev.values():
            cudart.cudaFree(ptr)
        return out


def load_inputs(inputs_dir):
    inputs = {}
    for f in os.listdir(inputs_dir):
        if f.endswith(".npy"):
            inputs[f[:-4]] = np.load(os.path.join(inputs_dir, f))
    return inputs


def bench(engine_path, inputs_dir, num_runs=100, warmup=10):
    runner = Runner(engine_path)
    inputs = load_inputs(inputs_dir)
    for _ in range(warmup):
        runner.run(inputs)
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        runner.run(inputs)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"BENCH {os.path.basename(engine_path)} runs={num_runs} "
          f"mean={np.mean(times):.2f}ms min={np.min(times):.2f}ms max={np.max(times):.2f}ms")


def main():
    cmd = sys.argv[1]
    if cmd == "build":
        build_engine(sys.argv[2], sys.argv[3],
                     len(sys.argv) > 4 and sys.argv[4] == "strongly_typed")
    elif cmd == "bench":
        bench(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 100)
    else:
        raise SystemExit(f"unknown cmd {cmd}")


if __name__ == "__main__":
    main()
