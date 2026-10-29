"""Native TensorRT/CUDA policy. No Torch import on the engine-cache hot path."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from .common import PolicyRuntime

EXPORT_VERSION = 1


def dependencies():
    try:
        import tensorrt as trt
        try:
            from cuda.bindings import runtime as cuda
        except ImportError:
            from cuda import cudart as cuda
    except ImportError as exc:
        raise RuntimeError('TensorRT backend needs its optional dependencies: pip install -e ".[tensorrt]". '
                           'Use backend: eager explicitly if TensorRT is unavailable.') from exc
    return trt, cuda


def checked(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f'CUDA failed: {error}')
    return values[0] if len(values) == 1 else tuple(values)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash():
    root = Path(__file__).resolve().parents[1]
    files = [*Path(__file__).parent.glob('*.py'), root / 'trainer/onnx.py',
             *sorted((root / 'models/policy/flow_matching').rglob('*.py')),
             root / 'common/inference.py', root / 'models/base_runner.py']
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cache_identity(checkpoint, options, hardware):
    return dict(export_version=EXPORT_VERSION,
                config_sha256=file_hash(checkpoint / 'config.json'),
                weights_sha256=file_hash(checkpoint / 'model.safetensors'),
                source_sha256=source_hash(), precision=options['precision'],
                batch_size=options['tensorrt_batch_size'], use_kv_cache=options['use_kv_cache'],
                hardware=hardware)


def ensure_engines(checkpoint, options, hardware, device):
    """Content-addressed startup build. File lock + manifest-last atomic publish."""
    identity = cache_identity(checkpoint, options, hardware)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(options['tensorrt_engine_dir'] or os.environ.get(
        'S1_TENSORRT_CACHE', str(Path.cwd() / '.cache/100hzs1/tensorrt'))).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / key

    def ready():
        manifest = directory / 'manifest.json'
        if not manifest.is_file():
            return None
        saved = json.loads(manifest.read_text())
        if saved.get('identity') != identity:
            raise RuntimeError(f'TensorRT cache identity mismatch: {directory}')
        for name, digest in saved['engine_sha256'].items():
            path = directory / name
            if not path.is_file() or file_hash(path) != digest:
                raise RuntimeError(f'TensorRT cache is incomplete or modified: {path}; remove this cache entry and rebuild')
        return saved

    with (root / f'{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = ready()
        if saved is not None:
            return directory, saved['metadata']
        temporary = Path(tempfile.mkdtemp(prefix=f'.{key}-', dir=root))
        try:
            print(f'[100HzS1] Building TensorRT {options["precision"]} engines before control starts: {directory}', flush=True)
            command = [sys.executable, '-m', '100hzs1.inference.export_tensorrt',
                       '--checkpoint', str(checkpoint), '--output', str(temporary), '--device', str(device),
                       '--precision', options['precision'], '--batch-size', str(options['tensorrt_batch_size']),
                       '--kv-cache' if options['use_kv_cache'] else '--no-kv-cache']
            # The child isolates Torch, ONNX patches and build-time allocations
            # from the native policy process. It exits before control begins.
            with (temporary / 'build.log').open('w') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                tail = '\n'.join((temporary / 'build.log').read_text(errors='replace').splitlines()[-35:])
                raise RuntimeError(f'TensorRT export/build failed ({result.returncode}):\n{tail}')
            if identity != cache_identity(checkpoint, options, hardware):
                raise RuntimeError('Checkpoint or source changed during TensorRT build; retry with stable files')
            metadata = json.loads((temporary / 'metadata.json').read_text())
            manifest = dict(identity=identity, metadata=metadata,
                            engine_sha256={p.name: file_hash(p) for p in temporary.glob('*.plan')})
            if set(manifest['engine_sha256']) != {'encoder.plan', 'step.plan', 'post.plan'}:
                raise RuntimeError('TensorRT builder did not produce all policy stages')
            (temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2))
            # Only successfully built and verified bundles become visible.
            if directory.exists():
                raise RuntimeError(f'Incomplete TensorRT cache entry already exists: {directory}')
            temporary.rename(directory)
            return directory, metadata
        finally:
            if temporary.exists():
                # Failed build logs are retained for diagnosis; no weights are
                # stored here, and these are exclusively our temporary files.
                failed = root / f'{key}.last-build.log'
                if (temporary / 'build.log').is_file():
                    shutil.copyfile(temporary / 'build.log', failed)
                shutil.rmtree(temporary)


class Buffer:
    def __init__(self, cuda, shape, dtype=np.float32):
        self.cuda = cuda
        self.shape, self.dtype = tuple(int(v) for v in shape), np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        self.ptr = int(checked(cuda.cudaMalloc(max(1, self.nbytes))))

    def put(self, value):
        value = np.ascontiguousarray(value, dtype=self.dtype)
        if value.shape != self.shape:
            raise ValueError(f'TensorRT input shape {value.shape}; engine requires {self.shape}')
        checked(self.cuda.cudaMemcpy(self.ptr, value.ctypes.data, self.nbytes,
                                    self.cuda.cudaMemcpyKind.cudaMemcpyHostToDevice))

    def get(self):
        value = np.empty(self.shape, dtype=self.dtype)
        checked(self.cuda.cudaMemcpy(value.ctypes.data, self.ptr, self.nbytes,
                                    self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return value

    def close(self):
        if self.ptr:
            checked(self.cuda.cudaFree(self.ptr))
            self.ptr = 0

    def __del__(self):
        try:
            if getattr(self, 'ptr', 0):
                self.close()
        except Exception:
            pass


class Engine:
    def __init__(self, path, batch, trt, cuda):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f'TensorRT cannot deserialize {path}; rebuild for this GPU/TensorRT version')
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.buffers = {}, {}, []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = tuple(batch if v < 0 else v for v in self.engine.get_tensor_shape(name))
                if -1 in self.engine.get_tensor_shape(name) and not self.context.set_input_shape(name, shape):
                    raise RuntimeError(f'Cannot set TensorRT input {name}={shape}')
                buffer = Buffer(cuda, shape, trt.nptype(self.engine.get_tensor_dtype(name)))
                self.inputs[name] = buffer
                self.buffers.append(buffer)
        if self.context.infer_shapes():
            raise RuntimeError(f'Incomplete TensorRT input shapes: {path}')
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                buffer = Buffer(cuda, self.context.get_tensor_shape(name), trt.nptype(self.engine.get_tensor_dtype(name)))
                self.outputs[name] = buffer
                self.buffers.append(buffer)

    def run(self, stream, inputs=None, outputs=None):
        for fields, bindings in [(self.inputs, inputs or self.inputs), (self.outputs, outputs or self.outputs)]:
            for key in fields:
                if not self.context.set_tensor_address(key, bindings[key].ptr):
                    raise RuntimeError(f'Cannot bind TensorRT tensor {key}')
        if not self.context.execute_async_v3(stream_handle=stream):
            raise RuntimeError('TensorRT enqueue failed')

    def close(self):
        for buffer in self.buffers:
            buffer.close()
        self.buffers.clear()
        self.context = self.engine = self.runtime = None

    def __del__(self):
        try:
            if hasattr(self, 'buffers'):
                self.close()
        except Exception:
            pass


class TensorRTPolicy(PolicyRuntime):
    def __init__(self, checkpoint, device, options):
        if str(device) != 'cuda' and not str(device).startswith('cuda:'):
            raise ValueError('TensorRT requires a CUDA device; select backend: eager for CPU')
        self.trt, self.cuda = dependencies()
        self.device_index = int(str(device).split(':', 1)[1]) if ':' in str(device) else 0
        checked(self.cuda.cudaSetDevice(self.device_index))
        props = checked(self.cuda.cudaGetDeviceProperties(self.device_index))
        name = props.name.decode() if isinstance(props.name, bytes) else str(props.name)
        hardware = dict(gpu=name, sm=[props.major, props.minor], tensorrt=self.trt.__version__,
                        cuda=checked(self.cuda.cudaRuntimeGetVersion()), driver=checked(self.cuda.cudaDriverGetVersion()))
        self.engine_dir, metadata = ensure_engines(Path(checkpoint), options, hardware, device)
        self._initialize(metadata, options, device)
        self.batch = options['tensorrt_batch_size']
        self._graph = self._graph_exec = None
        self._captures = self._replays = 0
        self._closed = False
        self._engines, self._buffers = [], []
        self.stream = checked(self.cuda.cudaStreamCreateWithFlags(self.cuda.cudaStreamNonBlocking))
        try:
            for scope in ['encoder', 'step', 'post']:
                engine = Engine(self.engine_dir / f'{scope}.plan', self.batch, self.trt, self.cuda)
                setattr(self, scope, engine)
                self._engines.append(engine)
            self.noise = self._allocate((self.batch, self.horizon, self.action_dim))
            self.base_state = self._allocate((self.batch, 1, self.action_dim))
            self.values = [self._allocate(self.noise.shape), self._allocate(self.noise.shape)]
            self.dt = self._allocate((self.batch, 1, 1))
            sign = -1 if self.flow_type == 'meanflow' else 1
            self.dt.put(np.full(self.dt.shape, sign / self.num_inference_timesteps, dtype=np.float32))
            self.times, self.targets = [], []
            for i in range(self.num_inference_timesteps):
                if self.flow_type == 'meanflow':
                    t = (1 - i / self.num_inference_timesteps) * metadata['time_scale']
                    r = (1 - (i + 1) / self.num_inference_timesteps) * metadata['time_scale']
                else:
                    t = int(i / self.num_inference_timesteps * metadata['num_timestep_buckets'])
                    r = 0
                for value, destination in [(t, self.times), (r, self.targets)]:
                    buffer = self._allocate((self.batch,))
                    buffer.put(np.full(buffer.shape, value, dtype=np.float32))
                    destination.append(buffer)
        except Exception:
            self.close()
            raise

    def _allocate(self, shape):
        buffer = Buffer(self.cuda, shape)
        self._buffers.append(buffer)
        return buffer

    @property
    def graph_stats(self):
        return dict(graph_captures=self._captures, graph_replays=self._replays, graph_scope='full_pipeline')

    def _full(self, return_flow=False):
        checked(self.cuda.cudaMemcpyAsync(self.values[0].ptr, self.noise.ptr, self.noise.nbytes,
                self.cuda.cudaMemcpyKind.cudaMemcpyDeviceToDevice, self.stream))
        self.encoder.run(self.stream)
        flow = []
        if return_flow:
            checked(self.cuda.cudaStreamSynchronize(self.stream))
            initial = self.values[0].get()
            initial[..., self.metadata['static_joint_mask']] = 0
            flow.append(initial[:, self.n_obs_steps:self.n_obs_steps+self.n_action_steps])
        for i, (t, r) in enumerate(zip(self.times, self.targets)):
            condition = self.encoder.outputs['kv' if self.inference_options['use_kv_cache'] else 'scene']
            bindings = dict(noisy=self.values[i % 2], dt=self.dt, timestep=t, target_time=r,
                            condition=condition, **self.encoder.outputs)
            self.step.run(self.stream, bindings, {'next_noisy': self.values[(i+1) % 2]})
            if return_flow:
                checked(self.cuda.cudaStreamSynchronize(self.stream))
                flow.append(self.values[(i+1) % 2].get()[:, self.n_obs_steps:self.n_obs_steps+self.n_action_steps])
        self.post.run(self.stream, dict(noisy=self.values[self.num_inference_timesteps % 2],
                       normalized=self.encoder.outputs['normalized'], base_state=self.base_state))
        return flow

    def _capture(self):
        for _ in range(3):
            self._full()
        checked(self.cuda.cudaStreamSynchronize(self.stream))
        checked(self.cuda.cudaStreamBeginCapture(self.stream,
                self.cuda.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
        # Capture errors are surfaced; never switch to eager after a failure.
        self._full()
        self._graph = checked(self.cuda.cudaStreamEndCapture(self.stream))
        self._graph_exec = checked(self.cuda.cudaGraphInstantiate(self._graph, 0))
        self._captures += 1

    def predict_action(self, observations, *, seed=None, noise_trajectory=None,
                       return_flow=False, use_cuda_graph=None, use_kv_cache=None):
        with self._lock:
            if self._closed:
                raise RuntimeError('TensorRT policy is closed')
            checked(self.cuda.cudaSetDevice(self.device_index))
            graph = self.inference_options['use_cuda_graph'] if use_cuda_graph is None else use_cuda_graph
            if use_kv_cache is not None and use_kv_cache != self.inference_options['use_kv_cache']:
                raise ValueError('K/V mode is fixed at engine build time; load a policy with the requested use_kv_cache')
            if graph and return_flow:
                raise ValueError('return_flow requires use_cuda_graph=False')
            arrays, batch = self._inputs(observations)
            if batch != self.batch:
                raise ValueError(f'TensorRT engine batch={self.batch}, received {batch}; configure tensorrt_batch_size')
            if 'images' in self.encoder.inputs:
                self.encoder.inputs['images'].put(np.stack([arrays[k] for k in self.rgb_keys], axis=2))
            if 'state' in self.encoder.inputs:
                self.encoder.inputs['state'].put(arrays[self.state_key])
            key = self.state_key.replace('delta_', '')
            self.base_state.put(arrays[key][:, :1])
            self.noise.put(self._noise(batch, seed, noise_trajectory))
            flow = None
            if graph:
                if self._graph_exec is None:
                    self._capture()
                checked(self.cuda.cudaGraphLaunch(self._graph_exec, self.stream))
                self._replays += 1
            else:
                flow = self._full(return_flow)
            checked(self.cuda.cudaStreamSynchronize(self.stream))
            result = {k: buffer.get() for k, buffer in self.post.outputs.items()}
            if return_flow:
                result['flow'] = flow
                result['gt'] = self.encoder.outputs['normalized'].get()[:, self.n_obs_steps:self.n_obs_steps+self.n_action_steps]
            self.last_nfe = self.num_inference_timesteps
            self.network_calls += self.last_nfe
            return self._validate_output(result, batch)

    def close(self):
        if getattr(self, '_closed', True):
            return
        checked(self.cuda.cudaSetDevice(self.device_index))
        checked(self.cuda.cudaStreamSynchronize(self.stream))
        if self._graph_exec is not None:
            checked(self.cuda.cudaGraphExecDestroy(self._graph_exec))
        if self._graph is not None:
            checked(self.cuda.cudaGraphDestroy(self._graph))
        for engine in self._engines:
            engine.close()
        for buffer in self._buffers:
            buffer.close()
        checked(self.cuda.cudaStreamDestroy(self.stream))
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
