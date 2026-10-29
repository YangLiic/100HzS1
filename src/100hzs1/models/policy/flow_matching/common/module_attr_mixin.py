import torch
from torch import nn

class ModuleAttrMixin(nn.Module):
    """Device/dtype tracking without an empty parameter in checkpoint weights."""
    def __init__(self):
        super().__init__()
        self.register_buffer("_device_dtype_anchor", torch.empty(0), persistent=False)

    @property
    def device(self):
        return next(self.parameters(), self._device_dtype_anchor).device

    @property
    def dtype(self):
        return next(self.parameters(), self._device_dtype_anchor).dtype
