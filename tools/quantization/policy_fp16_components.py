"""Export, validate and time B0 policy components with native TensorRT only."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    diff = a-b
    norm = max(float(np.linalg.norm(b)), 1e-12)
    return dict(rmse=float(np.sqrt(np.mean(diff**2))),
                relative_rmse=float(np.linalg.norm(diff)/norm),
                max_abs=float(np.abs(diff).max()), finite=True)


def export_case(plan, case, out):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(42)
    exp = importlib.import_module('100hzs1.inference.export_tensorrt')
    p = importlib.import_module('100hzs1.models.base_runner').load_policy(case['checkpoint'], device='cpu').float().eval()
    p.num_inference_timesteps = 4
    p.model.dtype = p.model.t_embedder.dtype = torch.float32
    assert p.model.r_embedder is None and p.flow_type == 'fm'
    assert p.config.hidden_size == case['dimension'] and p.config.visual_pooling is case['pooling']
    assert p.config.visual_variant == case['variant'] and len(p.obs_encoder.rgb_key) == 3
    assert p.model.time_injection == 'tau_token'
    meta = exp.metadata_for_export(p, True)
    meta.update(precision='fp16', precision_recipe='fp16_sensitive_reference_ops_fp32_v1')
    scene = p.obs_encoder.scene_encoder
    grid = (2, 3) if case['variant'] == 'stage4_rms' else (8, 10)
    expected_tokens = 12 if case['pooling'] else 18
    assert sum(isinstance(m, nn.LayerNorm) for m in scene.mv_fuser.modules()) == 5
    assert len(scene.model_adapter.downsample_blocks) == (0 if case['variant'] == 'stage4_rms' else 2)
    if case['variant'] == 'stage4_rms':
        assert type(p.scene_adaptor.norm).__name__ == 'RmsNorm'
    prep = exp.Preprocess(meta)

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__(); self.backbone = scene.backbone
        def forward(self, processed):
            return self.backbone(processed)['image_tokens']

    class Visual(nn.Module):
        def __init__(self):
            super().__init__(); self.p = p; self.preprocess = prep
        def forward(self, images):
            s = self.p.obs_encoder.scene_encoder
            tokens = s.backbone(self.preprocess(images))
            tokens['image_tokens'] = s.backbone_projection(tokens['image_tokens'])
            x = s.token_forward(tokens).reshape(1, -1, s.dim_out)
            return self.p.adapt_conditions(x)

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__(); self.adapter = scene.model_adapter
        def forward(self, spatial):
            x = self.adapter.downsample_blocks(spatial)
            x = self.adapter.patch_merging(x, self.adapter.shape_out)
            return x.reshape(1, -1, x.shape[-1])

    class DiTOnly(nn.Module):
        def __init__(self):
            super().__init__(); self.model = p.model
        def forward(self, embedded, timestep, kv):
            cache = tuple((kv[:, i, 0], kv[:, i, 1]) for i in range(len(self.model.blocks)))
            return self.model(embedded, timestep, None, kv_cache=cache)

    modules = dict(backbone=Backbone().eval(), visual=Visual().eval(),
                   adapter=Adapter().eval(), dit=DiTOnly().eval(),
                   encoder=exp.Encoder(p, meta, True).eval(), step=exp.Step(p, True).eval(),
                   post=exp.Post(p).eval())
    source = Path(plan['input_root'])
    images = np.load(source/'eval_images.npy', mmap_mode='r')
    states = np.load(source/'eval_state.npy', mmap_mode='r')
    noises = np.load(source/'eval_noise.npy', mmap_mode='r')
    specs = None
    out.mkdir(parents=True, exist_ok=False)
    (out/'refs').mkdir(); (out/'bundle').mkdir()
    exp.patch_export()
    for sample, index in enumerate(plan['reference_indices']):
        rgb = torch.from_numpy(images[index:index+1].copy())[:, None]
        state = torch.from_numpy(states[index:index+1].copy())
        noise = torch.from_numpy(noises[index:index+1].copy())
        obs = {key: rgb[:, :, i] for i, key in enumerate(meta['rgb_keys'])}
        obs[meta['state_key']] = state
        calls = []
        hook = p.model.register_forward_pre_hook(
            lambda m, a, kw: calls.append((a[0].detach().clone(), a[1].detach().clone())), with_kwargs=True)
        try:
            with torch.inference_mode():
                full = p.predict_action(obs, noise_trajectory=noise, use_cuda_graph=False, use_kv_cache=True)
        finally:
            hook.remove()
        assert len(calls) == 4 and [int(t[0]) for _, t in calls] == [0, 250, 500, 750]
        with torch.inference_mode():
            processed = prep(rgb)
            original = scene.img_processing(rgb.reshape(-1, 3, 240, 320)).reshape_as(processed)
            torch.testing.assert_close(processed, original, rtol=2e-5, atol=2e-5)
            visual = modules['visual'](rgb)
            torch.testing.assert_close(visual, p.encode_image_conditions(obs, 1), rtol=2e-4, atol=1e-4)
            assert list(visual.shape) == [1, expected_tokens, case['dimension']]
            backbone = modules['backbone'](processed)
            fused = scene.mv_fuser(backbone)
            spatial = fused.reshape(3, *grid, 128).permute(0, 3, 1, 2).contiguous()
            adapter = modules['adapter'](spatial)
            kv, normalized = modules['encoder'](rgb, state)
            dt = torch.full((1, 1, 1), .25)
            t = torch.zeros(1); base = state[:, :1]
            step = modules['step'](noise, normalized, kv, t, t, dt)
            post = modules['post'](noise, normalized, base)
            data = dict(images=rgb, state=state, noise=noise, processed=processed,
                        backbone=backbone, visual=visual, spatial=spatial, adapter=adapter,
                        kv=kv, normalized=normalized, dt=dt, timestep=t, target_time=t,
                        base_state=base, step=step, post_action=post[0],
                        full_action=full['action'], full_predicted_raw=full['predicted_raw'])
            for i, (x, timestep) in enumerate(calls):
                data[f'dit_x{i}'] = x
                data[f'dit_t{i}'] = timestep.float()
                data[f'dit_y{i}'] = modules['dit'](x, timestep.float(), kv)
            np.savez(out/'refs'/f'sample_{sample}.npz',
                     **{k: v.detach().cpu().numpy() for k, v in data.items()})
            if specs is None:
                specs = [
                    ('backbone', [processed], ['processed'], ['tokens']),
                    ('visual', [rgb], ['images'], ['conditions']),
                    ('adapter', [spatial], ['spatial'], ['features']),
                    ('dit', [calls[0][0], t, kv], ['embedded', 'timestep', 'kv'], ['velocity']),
                    ('encoder', [rgb, state], ['images', 'state'], ['kv', 'normalized']),
                    ('step', [noise, normalized, kv, t, t, dt],
                     ['noisy', 'normalized', 'condition', 'timestep', 'target_time', 'dt'], ['next_noisy']),
                    ('post', [noise, normalized, base], ['noisy', 'normalized', 'base_state'],
                     ['action', 'predicted_raw']),
                ]
    for name, values, inputs, outputs in specs:
        parent = out/'bundle' if name in ('encoder', 'step', 'post') else out
        path = parent/f'{name}.onnx'
        with torch.no_grad():
            torch.onnx.export(modules[name], tuple(values), str(path), opset_version=19,
                              dynamo=False, input_names=inputs, output_names=outputs,
                              do_constant_folding=True)
        print('EXPORTED', case['id'], name, path.stat().st_size, flush=True)
    write(out/'bundle/metadata.json', meta)
    write(out/'export_manifest.json', dict(case=case, conditions=expected_tokens,
          actual_time_buckets=[0, 250, 500, 750], fusion_ln_count=5,
          weights_sha256=digest(Path(case['checkpoint'])/'model.safetensors'),
          config_sha256=digest(Path(case['checkpoint'])/'config.json'),
          norm=p.config.norm, metadata=meta,
          exports={str(x.relative_to(out)):digest(x) for x in out.rglob('*.onnx')}))


def build_case(plan, case, out):
    from trt_precision import build
    export = read(out/'export_manifest.json')
    for relative, expected in export['exports'].items():
        path = out/relative
        assert digest(path) == expected
        build(path, path.with_suffix('.plan'), 'fp32' if path.stem == 'post' else 'fp16', True)
    write(out/'build_manifest.json', dict(exports=export,
          engines={str(x.relative_to(out)):digest(x) for x in out.rglob('*.plan')}))


def native_runtime(case, out):
    tr = importlib.import_module('100hzs1.inference.tensorrt')
    common = importlib.import_module('100hzs1.common.inference')
    opts = common.checkpoint_inference_options(case['checkpoint'], overrides=dict(
        backend='tensorrt', precision='fp16', num_inference_timesteps=4,
        use_cuda_graph=False, use_kv_cache=True, tensorrt_batch_size=1))
    meta = read(out/'bundle/metadata.json')
    original = tr.ensure_engines
    try:
        tr.ensure_engines = lambda *a: (out/'bundle', meta)
        runtime = tr.TensorRTPolicy(case['checkpoint'], 'cuda:0', opts)
    finally:
        tr.ensure_engines = original
    assert runtime.num_inference_timesteps == 4 and len(runtime.times) == 4
    assert 'torch' not in sys.modules
    return tr, runtime


def component_inputs(name, data, dit_index=0):
    if name == 'backbone': return {'processed':data['processed']}
    if name == 'visual': return {'images':data['images']}
    if name == 'adapter': return {'spatial':data['spatial']}
    if name == 'dit': return dict(embedded=data[f'dit_x{dit_index}'], timestep=data[f'dit_t{dit_index}'], kv=data['kv'])


def observations(data, runtime):
    obs = {key:data['images'][:, :, i] for i, key in enumerate(runtime.rgb_keys)}
    obs[runtime.state_key] = data['state']
    return obs


def parity_case(plan, case, out):
    tr, runtime = native_runtime(case, out)
    engines = {name: tr.Engine(out/f'{name}.plan', 1, runtime.trt, runtime.cuda)
               for name in ['backbone', 'visual', 'adapter', 'dit']}
    results = []
    try:
        for sample in range(len(plan['reference_indices'])):
            data = np.load(out/'refs'/f'sample_{sample}.npz')
            row = dict(sample=sample)
            for name, e in engines.items():
                inp = component_inputs(name, data)
                for key, buffer in e.inputs.items(): buffer.put(inp[key])
                e.run(runtime.stream)
                tr.checked(runtime.cuda.cudaStreamSynchronize(runtime.stream))
                y = next(iter(e.outputs.values())).get()
                row[name] = metrics(y, data['dit_y0'] if name == 'dit' else data[name])
            actual = runtime.predict_action(observations(data, runtime), noise_trajectory=data['noise'])
            row['action'] = metrics(actual['action'], data['full_action'])
            row['predicted_raw'] = metrics(actual['predicted_raw'], data['full_predicted_raw'])
            row['action_basic_tolerance'] = bool(row['action']['rmse'] <= .01 and row['action']['max_abs'] <= .05)
            results.append(row)
            print('PARITY', case['id'], json.dumps(row), flush=True)
        write(out/'parity.json', dict(results=results, all_finite=True,
              action_basic_tolerance=all(r['action_basic_tolerance'] for r in results),
              definition='same real images/state and fixed FM noise; numerical comparison, not success-rate validation'))
    finally:
        for e in engines.values(): e.close()
        runtime.close()


def bench_case(plan, case, out):
    tr, runtime = native_runtime(case, out)
    native = importlib.import_module('benchmark_trt_latency')
    physical_gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    assert ',' not in physical_gpu
    native.pids = lambda: sorted(subprocess.check_output([
        'nvidia-smi', '-i', physical_gpu, '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).split())
    native.smi = lambda fields: subprocess.check_output([
        'nvidia-smi', '-i', physical_gpu, '--query-gpu='+fields,
        '--format=csv,noheader,nounits'], text=True).strip()
    assert 'torch' not in sys.modules
    engines = {name: tr.Engine(out/f'{name}.plan', 1, runtime.trt, runtime.cuda)
               for name in ['backbone', 'visual', 'adapter', 'dit']}
    data = np.load(out/'refs/sample_0.npz')
    for name, e in engines.items():
        inp = component_inputs(name, data)
        for key, buffer in e.inputs.items(): buffer.put(inp[key])
        e.run(runtime.stream)
    runtime.predict_action(observations(data, runtime), noise_trajectory=data['noise'])
    dit_inputs = []
    for i in range(4):
        inp = component_inputs('dit', data, i)
        buffers = {key:tr.Buffer(runtime.cuda, value.shape, value.dtype) for key, value in inp.items()}
        for key, buffer in buffers.items(): buffer.put(inp[key])
        dit_inputs.append(buffers)
    try:
        expected = native.pids()
        def dit4():
            for buffers in dit_inputs: engines['dit'].run(runtime.stream, buffers)
        scopes = dict(backbone=lambda:engines['backbone'].run(runtime.stream),
                      visual=lambda:engines['visual'].run(runtime.stream),
                      adapter=lambda:engines['adapter'].run(runtime.stream),
                      dit_four_calls=dit4, encoder_with_kv=lambda:runtime.encoder.run(runtime.stream),
                      full_fm4=runtime._full)
        records = []
        for round_index in range(plan['rounds']):
            order = list(scopes)
            if round_index % 2: order.reverse()
            for name in order:
                native.exclusive(expected)
                timed = native.measure(scopes[name], runtime.stream, plan['repeats'], plan['warmup'])
                native.exclusive(expected)
                records.append(dict(scope=name, round=round_index, cuda_ms=timed['cuda'], wall_ms=timed['wall']))
                print('TIMED', case['id'], name, round_index, float(np.median(timed['cuda'])), flush=True)
                (out/'timings.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
        service = []
        obs = observations(data, runtime)
        for _ in range(20): runtime.predict_action(obs, noise_trajectory=data['noise'])
        for _ in range(100):
            before = time.perf_counter_ns()
            runtime.predict_action(obs, noise_trajectory=data['noise'])
            service.append((time.perf_counter_ns()-before)/1e6)
        summary = {}
        for name in scopes:
            rows = [r for r in records if r['scope'] == name]
            gpu = np.concatenate([r['cuda_ms'] for r in rows]);wall = np.concatenate([r['wall_ms'] for r in rows])
            summary[name] = dict(gpu_p50_ms=float(np.median(gpu)), gpu_p95_ms=float(np.percentile(gpu,95)),
                                 wall_p50_ms=float(np.median(wall)), samples=len(gpu))
        assert runtime._captures == runtime._replays == 0
        write(out/'latency.json', dict(case=case, scopes=summary,
              service_wall_p50_ms=float(np.median(service)), service_wall_p95_ms=float(np.percentile(service,95)),
              service_wall_samples_ms=service, nfe=4, actual_time_buckets=[0,250,500,750],
              batch=1, views=3, kv_cache=True, cuda_graph=False,
              scope_definition='GPU-input full_fm4 includes preprocess, visual, KV, state/action adaptor, DiT x4, Euler and FP32 post; service_wall also includes H2D/D2H and input validation',
              engine_hashes=read(out/'build_manifest.json')['engines'],
              gpu=native.smi('name,uuid,driver_version,clocks.current.sm,temperature.gpu'),
              tensorrt=runtime.trt.__version__, torch_loaded=False))
    finally:
        for group in dit_inputs:
            for buffer in group.values(): buffer.close()
        for e in engines.values(): e.close()
        runtime.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--mode', choices=['export', 'build', 'parity', 'bench'], required=True)
    args = ap.parse_args();plan=read(args.plan)
    case = next(c for c in plan['cases'] if c['id'] == args.case)
    out = Path(plan['artifacts'])/case['id']
    funcs = dict(export=export_case, build=build_case, parity=parity_case, bench=bench_case)
    funcs[args.mode](plan, case, out)


if __name__ == '__main__':
    main()
