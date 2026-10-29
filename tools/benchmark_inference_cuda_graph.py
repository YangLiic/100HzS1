"""Validate existing checkpoints and time inference-only visual CUDA Graphs."""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_vfa_policy import compute_pids, load_inputs, load_models, measure, smi
from safetensors.torch import load_model
from vfa.models.base_runner import load_policy


def compare(expected, actual):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    return dict(max_abs_error=float((actual.float() - expected.float()).abs().max()),
                bitwise_equal=torch.equal(actual, expected), shape=list(actual.shape))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--include-eager', action='store_true',
                        help='Explicitly opt in to an eager comparison; default measures Graph only')
    parser.add_argument('--checkpoint', action='append', default=[], metavar='NAME=PATH',
                        help='Use actual checkpoints instead of the historical structural comparison')
    parser.add_argument('--dataset', type=Path,
                        help='Override the offline single-task dataset root')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=50)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    assert not compute_pids(), 'GPU in use; do not overlap external jobs'
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.checkpoint:
        models, details = {}, {}
        for value in args.checkpoint:
            name, separator, path = value.partition('=')
            if not separator or not name or name in models:
                raise ValueError('Each --checkpoint must be a unique NAME=PATH')
            path = Path(path)
            model = load_policy(str(path), device='cpu').eval()
            load_model(model, str(path/'model.safetensors'), strict=True, device='cpu')
            assert (model.n_obs_steps, model.n_image_obs_steps, model.horizon, model.n_action_steps) == (1,1,24,23)
            model.num_inference_timesteps = 10
            models[name] = model.cuda().eval()
            details[name] = dict(checkpoint=str(path), strict_reload=True,
                                fusion_width=model.obs_encoder.scene_encoder.dim_2d)
    else:
        models, details = load_models()
    if args.dataset is not None:
        # The shared input helper resolves this module constant at call time.
        import benchmark_vfa_policy
        benchmark_vfa_policy.DATASET = args.dataset
    expected_pids = compute_pids()
    assert len(expected_pids) == 1
    gpu, host, _, _, data = load_inputs(Path(__file__).resolve().parents[1], models, [1, 2])
    observations = [{k: v[i:i+1] for k, v in gpu[2].items()} for i in range(2)]
    environment = dict(gpu=smi('name,uuid,driver_version,power.limit'), torch=torch.__version__,
                       cuda=torch.version.cuda, precision='FP32 weights with BF16 autocast',
                       tf32=False, batch_definition='one observation = three RGB views',
                       timing_excludes_video_decode=True, capture_scope='visual backbone only',
                       kv_cache=True, include_eager=args.include_eager, rounds=args.rounds, repeats=args.repeats,
                       models=details, data=data)
    checks, records = [], []
    if args.include_eager:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for name, model in models.items():
                assert inspect.signature(model.predict_action).parameters['use_cuda_graph'].default is True
                scene = model.obs_encoder.scene_encoder
                keys = list(model.state_dict())
                for index, obs in enumerate(observations):
                    eager_features = model.encode_image_conditions(obs, 1)
                    graph_features = model.encode_image_conditions(obs, 1, use_cuda_graph=True)
                    feature_check = compare(eager_features, graph_features)
                    torch.manual_seed(42)
                    eager = model.predict_action(obs, use_cuda_graph=False)
                    torch.manual_seed(42)
                    graph = model.predict_action(obs)  # Exercise the DEFAULT.
                    row = dict(model=name, observation=index, features=feature_check,
                               action=compare(eager['action'], graph['action']),
                               raw=compare(eager['predicted_raw'], graph['predicted_raw']))
                    assert scene._backbone_cuda_graph.captures == 1
                    checks.append(row)
                    print('CHECK '+json.dumps(row), flush=True)
                # First call must not consume extra RNG because of warmup/capture.
                scene.clear_cuda_graph_cache()
                torch.manual_seed(142)
                eager = model.predict_action(observations[0], use_cuda_graph=False)['action']
                torch.manual_seed(142)
                started = time.perf_counter()
                first_graph = model.predict_action(observations[0])['action']
                torch.cuda.synchronize()
                checks.append(dict(model=name, fresh_capture_action=compare(eager, first_graph),
                                   cold_capture_full_prediction_ms=(time.perf_counter()-started)*1000))
                # Shape changes recapture; only one signature is retained.
                torch.manual_seed(42)
                eager = model.predict_action(gpu[2], use_cuda_graph=False)['action']
                torch.manual_seed(42)
                graph = model.predict_action(gpu[2])['action']
                checks.append(dict(model=name, batch2_action=compare(eager, graph)))
                assert scene._backbone_cuda_graph.captures == 2
                assert keys == list(model.state_dict())
                torch.manual_seed(42)
                a = model.predict_action(gpu[1], return_flow=True, use_cuda_graph=False)
                torch.manual_seed(42)
                b = model.predict_action(gpu[1], return_flow=True)
                compare(a['action'], b['action'])
                for x, y in zip(a['flow'], b['flow']): np.testing.assert_array_equal(x, y)
                scene.clear_cuda_graph_cache()
    else:
        # Graph-only invariants: actual replay, finite outputs, repeatability and
        # independent output ownership. No eager prediction or timing is run.
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for name, model in models.items():
                assert inspect.signature(model.predict_action).parameters['use_cuda_graph'].default is True
                scene = model.obs_encoder.scene_encoder
                keys = list(model.state_dict())
                for index, obs in enumerate(observations):
                    torch.manual_seed(42)
                    first = model.predict_action(obs)['action']
                    snapshot = first.clone()
                    torch.manual_seed(42)
                    repeated = model.predict_action(obs)['action']
                    result = compare(snapshot, repeated)
                    assert torch.isfinite(repeated).all()
                    torch.testing.assert_close(first, snapshot, rtol=0, atol=0)
                    runner = scene._backbone_cuda_graph
                    assert runner is not None and runner.captures == 1 and runner.replays >= 2
                    checks.append(dict(model=name, observation=index,
                                       graph_repeatability=result,
                                       captures=runner.captures, replays=runner.replays))
                assert keys == list(model.state_dict())
                scene.clear_cuda_graph_cache()
    cases = []
    for name, model in models.items():
        for enabled in ((False, True) if args.include_eager else (True,)):
            cases.extend([
                (name, 'visual_encoder', enabled,
                 lambda m=model, e=enabled: m.encode_image_conditions(gpu[1], 1, use_cuda_graph=e)),
                (name, 'full_policy_gpu_inputs', enabled,
                 lambda m=model, e=enabled: m.predict_action(gpu[1], use_cuda_graph=e)),
                (name, 'full_policy_host_inputs', enabled,
                 lambda m=model, e=enabled: m.predict_action(host[1], use_cuda_graph=e)),
            ])
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for round_index in range(args.rounds):
            order = list(cases)
            if round_index % 2: order.reverse()
            for name, scope, enabled, fn in order:
                assert compute_pids() == expected_pids, 'GPU sharing detected; timing invalid'
                row = dict(model=name, scope=scope, use_cuda_graph=enabled, round=round_index)
                row.update(measure(fn, warmup=10, repeats=args.repeats))
                records.append(row)
                with (args.output/'measurements.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print('MEASURE '+json.dumps({k: v for k,v in row.items() if k not in ('wall','cuda')}), flush=True)
    grouped = defaultdict(list)
    for row in records: grouped[(row['model'], row['scope'], row['use_cuda_graph'])].append(row)
    aggregate = []
    for (name, scope, enabled), rows in grouped.items():
        values = [t for r in rows for t in r['wall']]
        aggregate.append(dict(model=name, scope=scope, use_cuda_graph=enabled,
                              samples=len(values), p50_ms=float(np.median(values)),
                              p95_ms=float(np.percentile(values,95)),
                              round_p50_ms=[r['wall_summary']['p50_ms'] for r in rows]))
    assert compute_pids() == expected_pids
    (args.output/'summary.json').write_text(json.dumps(dict(environment=environment,
                                        checks=checks, measurements=aggregate), indent=2))
    (args.output/'COMPLETED').write_text('CUDA Graph validation and requested latency measurements passed.\n')
    print('GRAPH_VALIDATION_COMPLETED',flush=True)


if __name__ == '__main__':
    main()
