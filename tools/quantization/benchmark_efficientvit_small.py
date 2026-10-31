"""B0/B1 spatial-trunk benchmark; reuse the existing native TensorRT harness."""
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
OLD = ROOT / 'work/trt_latency_20261029/artifacts'
MODELS = {'b0': 'efficientvit_b0.r224_in1k', 'b1': 'efficientvit_b1.r256_in1k'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def export(root, model):
    # Torch is confined to this subprocess, never imported by the latency worker.
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cls = importlib.import_module('100hzs1.models.policy.flow_matching.emvis.encoder.efficientvit_encoder').EfficientViTEncoder
    name = MODELS[model]
    weights = ROOT / 'pretrained' / name
    source = json.loads((weights / 'download_source.json').read_text())
    expected = source['files']['model.safetensors']['lfs']['sha256']
    assert digest(weights / 'model.safetensors') == expected
    backbone = cls(name, frozen=True)
    backbone.load_pretrained_model(weights)
    backbone.float().eval().cuda()

    class SpatialTokens(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone

        def forward(self, images):
            return self.backbone(images)['image_tokens']

    wrapper = SpatialTokens().eval()
    data = np.load(root / 'eval_processed.npy', mmap_mode='r')
    refs = []
    with torch.inference_mode():
        for sample in data:
            refs.append(wrapper(torch.from_numpy(sample[None].copy()).cuda()).float().cpu().numpy())
    dest = root / model
    dest.mkdir(exist_ok=True)
    reference = np.concatenate(refs)
    np.save(dest / 'eval_tokens.npy', reference)
    inputs = torch.from_numpy(np.asarray(data[:1]).copy()).cuda()
    with torch.no_grad():
        torch.onnx.export(wrapper, (inputs,), str(dest / 'backbone.onnx'),
            input_names=['images'], output_names=['tokens'], opset_version=19,
            dynamic_axes={'images': {0: 'batch'}, 'tokens': {0: 'batch'}},
            dynamo=False, do_constant_folding=True)
    record = dict(model=name, token_shape=list(reference.shape[1:]),
                  trunk_parameters=sum(p.numel() for p in backbone.parameters()),
                  weights_sha256=expected, source=source,
                  onnx_sha256=digest(dest / 'backbone.onnx'),
                  reference_sha256=digest(dest / 'eval_tokens.npy'), torch=torch.__version__)
    (dest / 'export.json').write_text(json.dumps(record, indent=2))
    print('EXPORTED', model, record['trunk_parameters'], record['token_shape'], flush=True)


def run(root):
    root.mkdir(parents=True, exist_ok=True)
    for filename in ['cal_processed.npy', 'eval_processed.npy', 'data_manifest.json']:
        target = root / filename
        if not target.exists():
            target.symlink_to(OLD / filename)
    for model in MODELS:
        command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--model', model]
        subprocess.run(command + ['--mode', 'export'], check=True)
        subprocess.run(command + ['--mode', 'build'], check=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--mode', 'bench'], check=True)


def build(root, model):
    import benchmark_trt_latency as native
    assert 'torch' not in sys.modules
    assert not native.pids(), 'Another GPU process is running'
    for precision in ['fp16', 'int8']:
        for batch in [1, 16]:
            native.build(root, model, 'backbone', precision, batch)


def bench(root):
    import benchmark_trt_latency as native
    assert 'torch' not in sys.modules, 'Timing must not import PyTorch'
    assert not native.pids(), 'Another GPU process is running'
    data = np.load(root / 'eval_processed.npy', mmap_mode='r')
    cases = [(m, p, b) for m in ['b0', 'b1', 'b2', 'dino']
             for p in ['fp16', 'int8'] for b in [1, 16]]
    records, checks = [], []
    expected_pids = None
    started = time.time()
    for round_index in range(3):
        # Rotate/reverse order to reduce a consistent temperature/order advantage.
        order = cases[round_index * 5:] + cases[:round_index * 5]
        if round_index == 1:
            order = order[::-1]
        for model, precision, batch in order:
            location = root if model in MODELS else OLD
            plan = native.engine_path(location, model, 'backbone', precision, batch)
            engine = native.Engine(plan, batch)
            if expected_pids is None:
                # NVML reports host PIDs, which differ from container os.getpid().
                expected_pids = native.pids()
                assert len(expected_pids) == 1, f'Expected only this worker: {expected_pids}'
            stream = native.cuda(native.cu.cudaStreamCreateWithFlags(native.cu.cudaStreamNonBlocking))
            try:
                if round_index == 0:
                    reference = np.load(location / model / 'eval_tokens.npy')
                    actual = []
                    for start in range(0, len(data), batch):
                        engine.inputs['images'].put(data[start:start+batch])
                        engine.run(stream)
                        native.cuda(native.cu.cudaStreamSynchronize(stream))
                        actual.append(engine.outputs['tokens'].get())
                    check = dict(model=model, precision=precision, batch=batch,
                                 metrics=native.metrics(np.concatenate(actual), reference))
                    checks.append(check)
                    print('FEATURE_CHECK', check, flush=True)
                engine.inputs['images'].put(data[:batch])
                native.cuda(native.cu.cudaStreamSynchronize(stream))
                native.exclusive(expected_pids)
                before = native.smi('clocks.current.sm,temperature.gpu,power.draw')
                timing = native.measure(lambda: engine.run(stream), stream, repeats=100, warmup=30)
                native.exclusive(expected_pids)
                row = dict(model=model, precision=precision, batch=batch, round=round_index,
                           graph=False, scope='backbone', wall_ms=timing['wall'], cuda_ms=timing['cuda'],
                           plan=str(plan), plan_sha256=digest(plan), gpu_before=before)
                records.append(row)
                (root / 'measurements.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in records))
                print('MEASURED', model, precision, batch, round_index,
                      'wall_p50/p95', np.percentile(row['wall_ms'], [50, 95]).tolist(), flush=True)
            finally:
                native.cuda(native.cu.cudaStreamSynchronize(stream))
                engine.close()
                native.cuda(native.cu.cudaStreamDestroy(stream))
    assert len(records) == 48 and len(checks) == 16
    assert 'torch' not in sys.modules
    summary = []
    for model, precision, batch in cases:
        rows = [v for v in records if (v['model'], v['precision'], v['batch']) == (model, precision, batch)]
        values = np.concatenate([v['wall_ms'] for v in rows])
        gpu = np.concatenate([v['cuda_ms'] for v in rows])
        check = next(v for v in checks if (v['model'], v['precision'], v['batch']) == (model, precision, batch))
        summary.append(dict(model=model, precision=precision, batch=batch,
            wall_p50=float(np.percentile(values, 50)), wall_p95=float(np.percentile(values, 95)),
            cuda_p50=float(np.percentile(gpu, 50)), cuda_p95=float(np.percentile(gpu, 95)),
            feature_metrics=check['metrics']))
    result = dict(summary=summary, checks=checks, gpu=native.smi('name,driver_version,power.limit'),
                  tensorrt=native.trt.__version__, elapsed_seconds=time.time()-started,
                  script_sha256=digest(__file__), native_helper_sha256=digest(native.__file__),
                  processed_data_sha256=digest(root / 'eval_processed.npy'), torch_loaded=False,
                  input_shape='[BS,3 views,3,256,320]', excludes=['preprocessing','H2D/D2H','fusion','policy'],
                  rounds=3, warmup=30, repeats=100, cuda_graph=False, tf32=False)
    (root / 'summary.json').write_text(json.dumps(result, indent=2))
    (root / 'COMPLETE').write_text('48 timing rounds and 16 feature checks complete\n')
    print('ALL_SMALL_BACKBONES_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', choices=list(MODELS))
    parser.add_argument('--mode', choices=['run', 'export', 'build', 'bench'], default='run')
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.mode in ['export', 'build'] and args.model is None:
        parser.error('--model required for export/build')
    if args.mode == 'export':
        export(args.root, args.model)
    elif args.mode == 'build':
        build(args.root, args.model)
    elif args.mode == 'bench':
        bench(args.root)
    else:
        run(args.root)
