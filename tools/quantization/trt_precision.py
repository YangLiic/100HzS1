"""TensorRT FP16 builder preserving precision-sensitive reference operations."""
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build(onnx_path, engine_path, precision='fp16', protect=True, shapes=None):
    import tensorrt as trt
    engine_path = Path(engine_path)
    assert not engine_path.exists(), engine_path
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    assert parser.parse_from_file(str(onnx_path)), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.clear_flag(trt.BuilderFlag.TF32)
    cfg.builder_optimization_level = 3
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if precision == 'fp16':
        cfg.set_flag(trt.BuilderFlag.FP16)
    if shapes:
        profile = builder.create_optimization_profile()
        for key, value in shapes.items():
            profile.set_shape(key, value, value, value)
        cfg.add_optimization_profile(profile)
    protected = []
    if precision == "fp16" and protect:
        from importlib import import_module
        guard = import_module("100hzs1.inference.fp16_precision")
        protected = guard.protect_fp16_sensitive_layers(net, cfg, trt)
    cache = engine_path.parent / 'tactics.cache'
    cfg.set_timing_cache(cfg.create_timing_cache(cache.read_bytes() if cache.exists() else b''), False)
    started = time.time()
    print('BUILD', engine_path, 'protected', len(protected), flush=True)
    plan = builder.build_serialized_network(net, cfg)
    assert plan is not None, engine_path
    engine_path.write_bytes(bytes(plan))
    cache.write_bytes(bytes(cfg.get_timing_cache().serialize()))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    inspector = engine.create_engine_inspector()
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    engine_path.with_suffix('.layers.json').write_text(json.dumps(layers, indent=2))
    record = dict(precision=precision, protected_layers=protected, tf32=False,
                  onnx_sha256=sha(onnx_path), engine_sha256=sha(engine_path),
                  build_seconds=time.time()-started, tensorrt=trt.__version__)
    engine_path.with_suffix('.build.json').write_text(json.dumps(record, indent=2))
    print('BUILT', engine_path, record['build_seconds'], flush=True)
    return record
