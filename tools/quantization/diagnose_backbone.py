"""Isolate TensorRT export and FP16 arithmetic from trained policy correctness."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def metric(actual, reference):
    a, b = actual.astype(np.float64), reference.astype(np.float64)
    assert a.shape == b.shape and np.isfinite(a).all()
    return dict(rmse=float(np.sqrt(np.mean((a-b)**2))),
                relative_rmse=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                cosine=float(np.dot(a.ravel(), b.ravel())/np.linalg.norm(a)/np.linalg.norm(b)),
                max_abs=float(np.abs(a-b).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--mode', choices=['build', 'measure'], required=True)
    args = ap.parse_args()
    old = ROOT/'tools/quantization/outputs/b0_stage4_views3_vs4_20261102_v1'
    args.root.mkdir(parents=True, exist_ok=True)
    if args.mode == 'build':
        import benchmark_trt_latency as native
        from trt_precision import build
        assert not native.pids()
        for precision, protect, name in [('fp32', False, 'fp32'), ('fp16', True, 'fp16_protected')]:
            build(old/'b0_stage4/backbone.onnx', args.root/f'{name}.plan', precision, protect,
                  shapes={'images': (1, 3, 3, 256, 320)})
        return
    import benchmark_trt_latency as native
    assert 'torch' not in sys.modules and not native.pids()
    stream = native.cuda(native.cu.cudaStreamCreateWithFlags(native.cu.cudaStreamNonBlocking))
    images = np.load(old/'images.npy')[:, :3]
    reference = np.load(old/'b0_stage4/reference_v3.npy')
    rows = []
    paths = [('fp32', args.root/'fp32.plan'), ('fp16_protected', args.root/'fp16_protected.plan'),
             ('fp16_original', old/'b0_stage4/backbone_fp16_bs3.plan')]
    try:
        for name, path in paths:
            e = native.Engine(path, 3)
            try:
                e.inputs['images'].put(images)
                e.run(stream)
                native.cuda(native.cu.cudaStreamSynchronize(stream))
                out = e.outputs['tokens'].get()
                np.save(args.root/f'{name}_tokens.npy', out)
                timing = native.measure(lambda: e.run(stream), stream, repeats=300, warmup=100)
                row = dict(mode=name, metrics=metric(out, reference),
                           cuda_p50_ms=float(np.median(timing['cuda'])),
                           wall_p50_ms=float(np.median(timing['wall'])))
                rows.append(row)
                print('DIAGNOSED', json.dumps(row), flush=True)
            finally:
                e.close()
    finally:
        native.cuda(native.cu.cudaStreamDestroy(stream))
    (args.root/'diagnosis.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
