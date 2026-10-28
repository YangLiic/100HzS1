"""Matched L4 policy/vision/training timing with real decoded offline observations."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import csv
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq
from safetensors.torch import load_model
import torch
import timm
import transformers

from vfa.models.base_runner import load_policy
from vfa.models.policy.vfa.dit_fm_policy import VFADitFlowMatching, VFADitFlowMatchingConfig


RUNS = Path('/DRoboticsAILab/users/licheng.yang/checkpoints/VFA/blocks_ranking_rgb')
CHECKPOINTS = {
    'dinov3_base': RUNS / 'dinov3cnnb_3v_obs1_h24_8gpu_L4_20261028/checkpoint-10000',
    'efficientvit_b2_projected': RUNS / 'efficientvit_b2_3v_obs1_h24_8gpu_L4_20261028/checkpoint-10000',
}
DATASET = Path('/DRoboticsAILab/vepfs_20260622/licheng.yang/datasets/robotwin2.0_lerobot_clean_joint14/blocks_ranking_rgb')


def smi(fields):
    return subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader'], text=True).strip()


def compute_pids():
    value = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    return sorted(line.strip() for line in value.splitlines() if line.strip())


def summary(values):
    return dict(p50_ms=float(np.percentile(values, 50)), p95_ms=float(np.percentile(values, 95)),
                mean_ms=float(np.mean(values)), min_ms=min(values), max_ms=max(values))


def measure(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeats)]
    wall, device = [], []
    for begin, end in events:
        started = time.perf_counter()
        begin.record()
        function()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - started) * 1000)
        device.append(begin.elapsed_time(end))
    return dict(wall=wall, cuda=device, wall_summary=summary(wall), cuda_summary=summary(device),
                incremental_peak_mib=(torch.cuda.max_memory_allocated() - baseline) / 1024**2)


def load_models():
    models, details = {}, {}
    for name, path in CHECKPOINTS.items():
        model = load_policy(str(path), device='cpu').eval()
        load_model(model, str(path / 'model.safetensors'), strict=True, device='cpu')
        assert (model.n_obs_steps, model.n_image_obs_steps, model.horizon, model.n_action_steps) == (1, 1, 24, 23)
        assert model.num_inference_timesteps == 10
        assert inspect.signature(model.predict_action).parameters['use_kv_cache'].default is True
        models[name] = model
        details[name] = dict(checkpoint=str(path), strict_reload=True,
                             config_sha256=hashlib.sha256((path / 'config.json').read_bytes()).hexdigest())
    projected = models['efficientvit_b2_projected']
    config = deepcopy(projected.config.to_dict())
    config['obs_encoder']['emvis_config'].update(efficientvit_fusion_mode='native', dim_2d=None,
                                                load_backbone_pretrain=False)
    native = VFADitFlowMatching(VFADitFlowMatchingConfig.from_dict(config)).eval()
    excluded = ('obs_encoder.scene_encoder.mv_fuser.', 'obs_encoder.scene_encoder.model_adapter.',
                'obs_encoder.scene_encoder.backbone_projection.')
    state = {k: v for k, v in projected.state_dict().items() if not k.startswith(excluded)}
    incompatible = native.load_state_dict(state, strict=False)
    assert not incompatible.unexpected_keys, incompatible
    assert all(k.startswith(excluded) for k in incompatible.missing_keys), incompatible
    models['efficientvit_b2_native'] = native
    details['efficientvit_b2_native'] = dict(
        source_checkpoint=details['efficientvit_b2_projected']['checkpoint'],
        speed_only_untrained_native_fusion_and_adapter=True,
        description='same trained backbone, scene/state adaptors, DiT and normalizer; new native-width fusion/compression')
    for name, model in models.items():
        # requires_grad is not serialized in a checkpoint. DINO's legacy loader
        # only freezes while loading pretraining, so restore the intended training
        # setting explicitly before comparing training-step costs with EfficientViT.
        scene = model.obs_encoder.scene_encoder
        scene.backbone.requires_grad_(scene.backbone_trainable)
        details[name].update(backbone_trainable=scene.backbone_trainable,
                             parameters=sum(p.numel() for p in model.parameters()),
                             trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                             vision_parameters=sum(p.numel() for p in model.obs_encoder.parameters()),
                             fusion_width=model.obs_encoder.scene_encoder.dim_2d)
        model.cuda().eval()
    return models, details


def load_inputs(root, models, batches):
    spec = importlib.util.spec_from_file_location('offline_eval', root / 'scripts/openloop_eval/evaluate.py')
    offline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(offline)
    info = json.loads((DATASET / 'meta/info.json').read_text())
    episode = 4
    location = dict(episode_index=episode, episode_chunk=episode // info['chunks_size'])
    length = pq.read_metadata(DATASET / info['data_path'].format(**location)).num_rows
    anchors = np.linspace(30, length - 24, max(batches), dtype=int).tolist()
    keys = models['dinov3_base'].obs_encoder.rgb_key
    assert all(m.obs_encoder.rgb_key == keys for m in models.values())
    states, actions, images, windows = offline.load_episode(DATASET, info, episode, keys, anchors, 1, 1, 23)
    observations = [offline.build_observation(states, images, window, keys, 'cpu', 'bgr') for window in windows]
    gpu, host, vision, train = {}, {}, {}, {}
    for batch in batches:
        host[batch] = {key: torch.cat([o[key] for o in observations[:batch]]).pin_memory() for key in observations[0]}
        gpu[batch] = {key: value.cuda() for key, value in host[batch].items()}
        vision[batch] = {key: gpu[batch][key][:, 0] for key in keys}
        train_obs = dict(gpu[batch])
        train_obs['state'] = torch.from_numpy(np.stack([states[a - 1:a + 23] for a in anchors[:batch]])).cuda()
        target = torch.from_numpy(np.stack([actions[a - 1:a + 23] for a in anchors[:batch]])).cuda()
        train[batch] = dict(obs=train_obs, action={'action': target})
    return gpu, host, vision, train, dict(dataset=str(DATASET), episode=episode, anchors=anchors,
                                         source_color='BGR converted once to RGB', keys=list(keys),
                                         image_chw=list(observations[0][keys[0]].shape[-3:]),
                                         timing_excludes_video_decode=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--train-repeats', type=int, default=20)
    parser.add_argument('--train-warmup', type=int, default=5)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    assert not compute_pids(), 'GPU has another compute process before benchmark allocation'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    environment = dict(device=smi('name,uuid,driver_version,power.limit,clocks.max.sm'),
                       torch=torch.__version__, cuda=torch.version.cuda, timm=timm.__version__,
                       transformers=transformers.__version__, python=platform.python_version(),
                       torch_threads=1, cpu_affinity=sorted(os.sched_getaffinity(0)),
                       precision='BF16 autocast; original checkpoint parameter dtypes', tf32=False,
                       compile=False, nfe=10, horizon=24, n_obs=1, n_image_obs=1,
                       batch_definition='one synchronized observation = three RGB camera images plus proprioception',
                       rounds=args.rounds, repeats_per_round=args.repeats, warmup_per_case=args.warmup,
                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    models, details = load_models()
    expected_pids = compute_pids()
    assert len(expected_pids) == 1, expected_pids
    gpu, host, vision, training, data_info = load_inputs(root, models, args.batch_sizes)
    (args.output / 'environment.json').write_text(json.dumps(dict(environment=environment, models=details, data=data_info), indent=2))
    checks = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for name, model in models.items():
            for batch in args.batch_sizes:
                counts = [0] * len(model.model.blocks)
                hooks = []
                for index, block in enumerate(model.model.blocks):
                    def hook(module, inputs, output, i=index): counts[i] += 1
                    hooks.append(block.cross_attn.kv.register_forward_hook(hook))
                outputs, calls = {}, {}
                try:
                    for enabled in (False, True):
                        counts[:] = [0] * len(counts)
                        torch.manual_seed(42)
                        outputs[enabled] = model.predict_action(gpu[batch], use_kv_cache=enabled)['action']
                        calls[str(enabled)] = list(counts)
                finally:
                    for hook in hooks: hook.remove()
                torch.testing.assert_close(outputs[True], outputs[False], atol=1e-5, rtol=1e-4)
                assert torch.isfinite(outputs[True]).all()
                checks.append(dict(model=name, batch=batch, max_abs_error=float((outputs[True] - outputs[False]).abs().max()),
                                   shape=list(outputs[True].shape), per_layer_kv_calls=calls))
                print('CHECK ' + json.dumps(checks[-1]), flush=True)
    (args.output / 'correctness.json').write_text(json.dumps(checks, indent=2))
    cases = []
    for batch in args.batch_sizes:
        for name, model in models.items():
            cases.append((name, batch, 'full_visual', None,
                          lambda m=model, b=batch: m.obs_encoder(vision[b])))
            for cache in (False, True):
                cases.append((name, batch, 'full_policy_gpu_inputs', cache,
                              lambda m=model, b=batch, enabled=cache: m.predict_action(gpu[b], use_kv_cache=enabled)))
            if batch == 1:
                cases.append((name, batch, 'full_policy_host_float_inputs', True,
                              lambda m=model, b=batch: m.predict_action(host[b], use_kv_cache=True)))
    records = []
    def save(record):
        records.append(record)
        with (args.output / 'measurements.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
        print('MEASURE ' + json.dumps({k:v for k,v in record.items() if k not in ('wall','cuda')}), flush=True)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for round_index in range(args.rounds):
            order = list(cases)
            if round_index % 2: order.reverse()
            elif round_index: order = order[len(order)//2:] + order[:len(order)//2]
            for name, batch, scope, cache, function in order:
                assert compute_pids() == expected_pids, 'GPU sharing detected; benchmark invalidated'
                row = dict(model=name, batch=batch, scope=scope, kv_cache=cache, round=round_index,
                           gpu_before=smi('temperature.gpu,clocks.current.sm,pstate,power.draw'))
                row.update(measure(function, args.warmup, args.repeats))
                row['observations_per_second'] = batch * 1000 / row['wall_summary']['p50_ms']
                save(row)
    if args.training:
        for name, model in models.items():
            model.train()
            # lr=0 measures the real update kernels without changing checkpoint weights.
            optimizer = torch.optim.AdamW(model.parameters(), lr=0, betas=(.95,.999), weight_decay=1e-6, eps=1e-8)
            for batch in args.batch_sizes:
                def step(m=model, b=batch, opt=optimizer):
                    opt.zero_grad(set_to_none=False)
                    with torch.autocast('cuda', dtype=torch.bfloat16): loss = m.compute_loss(training[b])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                    opt.step()
                for round_index in range(args.rounds):
                    assert compute_pids() == expected_pids, 'GPU sharing detected'
                    row = dict(model=name, batch=batch, scope='train_forward_backward_adamw', kv_cache=False,
                               round=round_index, lr=0, excludes=['data_loading','H2D','EMA','checkpoint_save'])
                    row.update(measure(step, args.train_warmup, args.train_repeats))
                    row['observations_per_second'] = batch * 1000 / row['wall_summary']['p50_ms']
                    save(row)
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            model.eval()
            gc.collect()
            torch.cuda.empty_cache()
    aggregate = []
    groups = {}
    for record in records:
        key = (record['model'],record['batch'],record['scope'],record['kv_cache'])
        groups.setdefault(key,[]).append(record)
    for (name,batch,scope,cache), rows in groups.items():
        wall = [v for r in rows for v in r['wall']]
        device = [v for r in rows for v in r['cuda']]
        row = dict(model=name,batch=batch,scope=scope,kv_cache=cache,samples=len(wall),
                   wall_p50_ms=summary(wall)['p50_ms'],wall_p95_ms=summary(wall)['p95_ms'],
                   cuda_p50_ms=summary(device)['p50_ms'],cuda_p95_ms=summary(device)['p95_ms'],
                   round_p50_ms=[r['wall_summary']['p50_ms'] for r in rows],
                   incremental_peak_mib=max(r['incremental_peak_mib'] for r in rows))
        row['observations_per_second'] = batch * 1000 / row['wall_p50_ms']
        aggregate.append(row)
    (args.output / 'summary.json').write_text(json.dumps(dict(environment=environment,models=details,data=data_info,
                                                            correctness=checks,measurements=aggregate),indent=2))
    with (args.output / 'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(aggregate[0]));writer.writeheader();writer.writerows(aggregate)
    (args.output / 'COMPLETED').write_text('All requested cases and correctness checks finished.\n')
    print('BENCHMARK_COMPLETED',flush=True)


if __name__ == '__main__': main()
