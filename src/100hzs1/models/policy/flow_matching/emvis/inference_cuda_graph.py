"""Bounded, non-serialized CUDA Graph runner for eval-only visual backbones."""
import logging
import threading
import weakref

import torch
from torch.utils._pytree import tree_map
from torch.utils._python_dispatch import TorchDispatchMode

logger = logging.getLogger(__name__)
_CAPTURE_LOCK = threading.Lock()


class _GraphPreflight(TorchDispatchMode):
    """Reject known host-sync/dynamic-output ops before CUDA capture starts."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        unsupported = {torch.ops.aten._local_scalar_dense.default,
                       torch.ops.aten.nonzero.default,
                       torch.ops.aten.masked_select.default}
        if func in unsupported:
            raise RuntimeError(f'{func} requires host synchronization or dynamic output')
        if func == torch.ops.aten._to_copy.default:
            device = kwargs.get('device')
            if device is not None and torch.device(device).type == 'cpu' and args[0].is_cuda:
                raise RuntimeError('CUDA-to-CPU copy is not graph safe')
        if func == torch.ops.aten.copy_.default and not args[0].is_cuda and args[1].is_cuda:
            raise RuntimeError('CUDA-to-CPU copy is not graph safe')
        return func(*args, **kwargs)


class BackboneCudaGraph:
    """Cache one shape/precision/stream; return owned outputs on every replay.

    The owner must discard this runner on train/device/load transitions. It is
    intentionally not an nn.Module, parameter, buffer or checkpoint setting.
    """

    def __init__(self, module):
        self.module_ref = weakref.ref(module)
        self._lock = threading.Lock()
        self._key = None
        self._failed_key = None
        self._graph = self._input = self._output = None
        self._capture_started = False
        self.captures = 0
        self.replays = 0

    @staticmethod
    def eligible(module, images):
        return (
            images.is_cuda and not module.training and not torch.is_grad_enabled()
            and not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export()
            and not torch.cuda.is_current_stream_capturing()
        )

    @staticmethod
    def signature(images):
        return (
            tuple(images.shape), tuple(images.stride()), images.dtype, images.device,
            torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'),
            torch.is_inference_mode_enabled(),
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
            torch.are_deterministic_algorithms_enabled(),
            torch.cuda.current_stream(images.device).cuda_stream,
        )

    def _capture(self, module, images):
        static_input = images.clone()
        stream = torch.cuda.Stream(device=images.device)
        current = torch.cuda.current_stream(images.device)
        stream.wait_stream(current)
        # Keep casts inside the graph, rather than reusing an autocast weight
        # cache whose tensors may be released when the caller exits autocast.
        with torch.autocast('cuda', enabled=torch.is_autocast_enabled('cuda'),
                            dtype=torch.get_autocast_dtype('cuda'), cache_enabled=False):
            try:
                with torch.cuda.stream(stream):
                    with _GraphPreflight():
                        module(static_input)
                    for _ in range(2):
                        module(static_input)
            finally:
                current.wait_stream(stream)
            current.synchronize()
            graph = torch.cuda.CUDAGraph()
            # Outer stream context restores the caller's stream even if
            # torch.cuda.graph.__exit__ itself raises on this PyTorch version.
            self._capture_started = True
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    static_output = module(static_input)
            self._capture_started = False
        current.wait_stream(stream)
        self._input, self._output, self._graph = static_input, static_output, graph
        self.captures += 1

    def __call__(self, images):
        module = self.module_ref()
        if module is None:
            raise RuntimeError('Visual backbone no longer exists')
        if not self.eligible(module, images):
            return module(images)
        with self._lock, torch.cuda.device(images.device):
            key = self.signature(images)
            if key == self._failed_key:
                return module(images)
            if key != self._key:
                # Bound memory even if inference repeatedly changes shape.
                self._graph = self._input = self._output = None
                self._key = self._failed_key = None
                self._capture_started = False
                try:
                    with _CAPTURE_LOCK:
                        self._capture(module, images)
                except RuntimeError as error:
                    if self._capture_started:
                        # Failed captures can corrupt allocator/RNG bookkeeping
                        # in PyTorch 2.7.1. Do not pretend eager is still safe.
                        raise RuntimeError(
                            'Visual CUDA Graph capture failed after capture began. '
                            'Restart inference with use_cuda_graph=False. '
                            'No model weights were modified.'
                        ) from error
                    self._failed_key = key
                    logger.warning('Visual CUDA Graph preparation failed; using eager for '
                                   'this input signature: %s', error)
                    return module(images)
                self._key = key
                logger.info('Visual CUDA Graph captured: shape=%s dtype=%s autocast=%s',
                            tuple(images.shape), images.dtype, key[4:6])
            self._input.copy_(images)
            self._graph.replay()
            self.replays += 1
            # The next replay overwrites graph-owned memory. Callers must get
            # independent tensors, including the aliased position metadata.
            return tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x,
                            self._output)
