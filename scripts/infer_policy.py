#!/usr/bin/env python3
"""Run the YAML-selected eager/TensorRT backend on saved RGB/state observations."""
import argparse
from importlib import import_module
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--observations', type=Path, required=True,
                    help='NPZ with camera [B,T,C,H,W] RGB and state [B,T,D] arrays')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inference-config', type=Path)
    ap.add_argument('--backend', choices=['eager', 'tensorrt'])
    ap.add_argument('--precision', choices=['fp16', 'bf16', 'fp32'])
    ap.add_argument('--cuda-graph', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--kv-cache', action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument('--nfe', type=int)
    ap.add_argument('--tensorrt-engine-dir', type=Path)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.observations, allow_pickle=False) as data:
        observations = {k: np.asarray(data[k]) for k in data.files}
    load = import_module('100hzs1.inference').load_inference_policy
    overrides = dict(backend=args.backend, precision=args.precision, use_cuda_graph=args.cuda_graph,
                     use_kv_cache=args.kv_cache, num_inference_timesteps=args.nfe,
                     tensorrt_engine_dir=args.tensorrt_engine_dir)
    with load(args.checkpoint, args.device, args.inference_config, overrides) as policy:
        result = policy.predict_action(observations, seed=args.seed)
        print(f'backend={policy.backend} precision={policy.precision} '
              f'cuda_graph={policy.inference_options["use_cuda_graph"]} '
              f'nfe={policy.last_nfe} action={result["action"].shape}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('wb') as handle:
            np.savez(handle, **result)


if __name__ == '__main__':
    main()
