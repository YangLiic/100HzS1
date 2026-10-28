"""One-batch CUDA prefetch for pinned CPU DataLoaders."""
from collections.abc import Mapping

import torch
from accelerate.utils import send_to_device


_EMPTY = object()


def _record_stream(value, stream):
    if torch.is_tensor(value):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, Mapping):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _record_stream(item, stream)


class CudaPrefetchIterator:
    """Copy one future batch on a side stream; preserve order and lifetime.

    The input iterator must yield pinned CPU tensors. Callers retain normal
    ownership of the current batch; only one future device batch is held here.
    PyTorch's pinned allocator tracks asynchronous source-copy completion.
    """

    def __init__(self, iterator, device):
        self.iterator = iterator
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("CUDA prefetch requires a CUDA device")
        self.stream = torch.cuda.Stream(device=self.device)
        self.next_batch = _EMPTY
        self._preload()

    def _preload(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = _EMPTY
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = send_to_device(batch, self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is _EMPTY:
            raise StopIteration
        current = torch.cuda.current_stream(self.device)
        # Wait for the current batch before enqueueing the next copy: compute
        # need not wait for that future transfer.
        current.wait_stream(self.stream)
        batch = self.next_batch
        _record_stream(batch, current)
        self._preload()
        return batch
