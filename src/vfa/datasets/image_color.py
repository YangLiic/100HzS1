"""Source color correction shared by the LeRobot and Lance readers."""

import torch


def apply_bgr_to_rgb(item, camera_keys):
    """Replace camera tensors with RGB copies; leave cached source tensors intact.

    Decoders return CHW or TCHW tensors even when source metadata describes HWC.
    The correction concerns source channel semantics, not the codec pixel format.
    """
    for key in camera_keys:
        if key not in item:
            continue
        image = item[key]
        if not isinstance(image, torch.Tensor) or image.ndim < 3 or image.shape[-3] != 3:
            raise ValueError(f"BGR conversion expects a CHW/TCHW RGB-sized tensor for {key!r}")
        item[key] = image.flip(-3)
