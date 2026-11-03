"""Compare V=3/4 (effective image batch B*V) in the B0 TensorRT encoder."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'b0_stage4'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def export(args):
    import torch
    from safetensors import safe_open
    from torch import nn
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cls = importlib.import_module(
        '100hzs1.models.policy.flow_matching.emvis.encoder.efficientvit_encoder'
    ).EfficientViTEncoder
    backbone = cls('efficientvit_b0.r224_in1k', frozen=True, visual_variant='stage4_rms')
    prefix = 'obs_encoder.scene_encoder.backbone.'
    weights = args.checkpoint / 'model.safetensors'
    with safe_open(weights, framework='pt', device='cpu') as f:
        state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    backbone.load_state_dict(state, strict=True)
    backbone.float().eval()

    class Tokens(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone

        def forward(self, images):
            return self.backbone(images)['image_tokens']

    model = Tokens().eval()
    source = np.load(args.images, mmap_mode='r')
    assert source.shape[1:] == (3, 3, 256, 320), source.shape
    # Four real preprocessed camera images; V=3 uses the identical first three.
    images = np.asarray(source[:2]).reshape(-1, 3, 256, 320)[:4].copy()[None]
    np.save(args.output / 'images.npy', images)
    dest = args.output / MODEL
    dest.mkdir()
    with torch.inference_mode():
        for views in (3, 4):
            x = torch.from_numpy(images[:, :views].copy())
            reference = model(x)
            torch.testing.assert_close(reference, backbone(x[0])['image_tokens'], rtol=0, atol=0)
            assert list(reference.shape) == [1, views, 6, 128]
            np.save(dest / f'reference_v{views}.npy', reference.numpy())
    with torch.no_grad():
        torch.onnx.export(
            model, (torch.from_numpy(images),), str(dest / 'backbone.onnx'),
            input_names=['images'], output_names=['tokens'], opset_version=19,
            dynamic_axes={'images': {1: 'views'}, 'tokens': {1: 'views'}},
            dynamo=False, do_constant_folding=True,
        )
    record = dict(checkpoint=str(args.checkpoint), weights_sha256=digest(weights),
                  config_sha256=digest(args.checkpoint / 'config.json'),
                  source_images=str(args.images), source_images_sha256=digest(args.images),
                  input_shape=[1, 'V', 3, 256, 320], effective_image_batch='B*V = V',
                  output_shape=[1, 'V', 6, 128], shared_stage4_extra_calls=2,
                  same_4d_and_5d_output=True, torch=torch.__version__,
                  onnx_sha256=digest(dest / 'backbone.onnx'))
    write(args.output / 'export.json', record)
    print('EXPORTED', json.dumps(record), flush=True)


def build(args):
    import benchmark_trt_latency as native
    assert 'torch' not in sys.modules
    assert not native.pids(), 'GPU is shared; leave the other process intact'
    for views in (3, 4):
        native.build(args.output, MODEL, 'backbone', 'fp16', views)


def bench(args):
    import benchmark_trt_latency as native
    assert 'torch' not in sys.modules
    assert not native.pids(), 'GPU is shared; leave the other process intact'
    native.cuda(native.cu.cudaFree(0))
    expected = native.pids()
    assert len(expected) == 1, expected
    stream = native.cuda(native.cu.cudaStreamCreateWithFlags(native.cu.cudaStreamNonBlocking))
    engines = {}
    checks = []
    records = []
    images = np.load(args.output / 'images.npy')
    try:
        for views in (3, 4):
            e = native.Engine(native.engine_path(args.output, MODEL, 'backbone', 'fp16', views), views)
            engines[views] = e
            assert e.inputs['images'].shape == (1, views, 3, 256, 320)
            assert e.outputs['tokens'].shape == (1, views, 6, 128)
            e.inputs['images'].put(images[:, :views])
            e.run(stream)
            native.cuda(native.cu.cudaStreamSynchronize(stream))
            actual = e.outputs['tokens'].get().astype(np.float64)
            reference = np.load(args.output / MODEL / f'reference_v{views}.npy').astype(np.float64)
            assert np.isfinite(actual).all()
            cosine = float(np.dot(actual.ravel(), reference.ravel()) /
                           (np.linalg.norm(actual) * np.linalg.norm(reference)))
            rel_rmse = float(np.sqrt(np.mean((actual-reference)**2)) /
                             max(np.sqrt(np.mean(reference**2)), 1e-12))
            checks.append(dict(views=views, input_shape=list(e.inputs['images'].shape),
                               output_shape=list(actual.shape), cosine=cosine,
                               relative_rmse=rel_rmse, max_abs=float(np.abs(actual-reference).max()),
                               finite=True,
                               within_reference_tolerance=bool(rel_rmse < .05 and cosine > .99)))
            print('NUMERICAL_CHECK', json.dumps(checks[-1]), flush=True)
        for round_index in range(args.rounds):
            # AB/BA balances order, frequency and temperature drift.
            order = (3, 4) if round_index % 2 == 0 else (4, 3)
            for views in order:
                native.exclusive(expected)
                before = native.smi('clocks.current.sm,temperature.gpu,power.draw,utilization.gpu')
                e = engines[views]
                # Bind once outside timing: compare enqueue + kernels, without H2D/D2H.
                def enqueue():
                    assert e.context.execute_async_v3(stream_handle=stream)
                timing = native.measure(enqueue, stream, repeats=args.repeats, warmup=args.warmup)
                native.exclusive(expected)
                row = dict(round=round_index, views=views, gpu_before=before,
                           cuda_ms=timing['cuda'], wall_ms=timing['wall'])
                records.append(row)
                (args.output / 'measurements.jsonl').write_text(
                    ''.join(json.dumps(x)+'\n' for x in records))
                print('MEASURED', views, round_index,
                      'CUDA_p50_ms', float(np.median(timing['cuda'])),
                      'wall_p50_ms', float(np.median(timing['wall'])), flush=True)
    finally:
        native.cuda(native.cu.cudaStreamSynchronize(stream))
        for e in engines.values():
            e.close()
        native.cuda(native.cu.cudaStreamDestroy(stream))
    summaries = []
    for views in (3, 4):
        rows = [x for x in records if x['views'] == views]
        gpu = np.concatenate([x['cuda_ms'] for x in rows])
        wall = np.concatenate([x['wall_ms'] for x in rows])
        summaries.append(dict(views=views, samples=len(gpu),
            cuda_p50_ms=float(np.median(gpu)), cuda_p95_ms=float(np.percentile(gpu, 95)),
            wall_p50_ms=float(np.median(wall)), wall_p95_ms=float(np.percentile(wall, 95)),
            per_image_cuda_p50_ms=float(np.median(gpu)/views),
            per_round_cuda_p50_ms=[float(np.median(x['cuda_ms'])) for x in rows]))
    result = dict(status='completed', scope='EfficientViT-B0 + two shared stage4 repeats',
        precision='TensorRT FP16 enabled, FP32 I/O', tf32=False, cuda_graph=False,
        input_shape='[1,V,3,256,320]', batch_is='effective image batch B*V=V',
        timing='CUDA events and synchronized perf_counter_ns, whole batch',
        excluded=['preprocessing', 'H2D/D2H', 'fusion', 'scene MLP', 'DiT'],
        gpu=native.smi('name,uuid,driver_version,power.limit'), tensorrt=native.trt.__version__,
        cuda_runtime=native.cuda(native.cu.cudaRuntimeGetVersion()),
        warmup=args.warmup, repeats=args.repeats, rounds=args.rounds,
        independent_static_profiles=True, torch_loaded=False, checks=checks, summary=summaries,
        export=json.loads((args.output/'export.json').read_text()),
        plans={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob('*.plan')},
        script_sha256=digest(__file__), helper_sha256=digest(native.__file__))
    result['all_within_reference_tolerance'] = all(x['within_reference_tolerance'] for x in checks)
    result['reference_tolerance'] = dict(relative_rmse_below=.05, cosine_above=.99)
    result['cuda_p50_ratio_v4_over_v3'] = summaries[1]['cuda_p50_ms']/summaries[0]['cuda_p50_ms']
    write(args.output/'summary.json', result)
    (args.output/'COMPLETE').write_text('V=3 and V=4 timing completed; inspect numerical checks before deployment\n')
    print('COMPLETED', json.dumps(summaries), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--images', type=Path, default=ROOT/'work/trt_latency_20261029/artifacts/eval_processed.npy')
    ap.add_argument('--mode', choices=['run', 'export', 'build', 'bench'], default='run')
    ap.add_argument('--rounds', type=int, default=6)
    ap.add_argument('--repeats', type=int, default=500)
    ap.add_argument('--warmup', type=int, default=200)
    args = ap.parse_args()
    for key in ['checkpoint', 'output', 'images']:
        setattr(args, key, getattr(args, key).resolve())
    assert args.rounds > 1 and args.repeats > 0 and args.warmup > 0
    if args.mode == 'run':
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output/'request.json', dict(checkpoint=str(args.checkpoint),
              images=str(args.images), views=[3, 4], rounds=args.rounds,
              repeats=args.repeats, warmup=args.warmup, started_at=time.time()))
        for mode in ['export', 'build', 'bench']:
            cmd = [sys.executable, str(Path(__file__).resolve()), '--mode', mode,
                   '--checkpoint', str(args.checkpoint), '--output', str(args.output),
                   '--images', str(args.images), '--rounds', str(args.rounds),
                   '--repeats', str(args.repeats), '--warmup', str(args.warmup)]
            subprocess.run(cmd, check=True)
    else:
        globals()[args.mode](args)


if __name__ == '__main__':
    main()
