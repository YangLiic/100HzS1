"""Dataset backend selector.

`train_dataset._target_` points here; `backend: lerobot|lance|lance_video` picks the
implementation. Both run in the same env (e.g. VFA): the lance backend
(vfa_dataset_lance.py) reads via lancedb/pylance directly and does not
import lerobot, so the two never conflict.
"""


def build_dataset(backend: str = "lerobot", **kwargs):
    if backend == "lerobot":
        from vfa.datasets.vfa_dataset import VFADataset

        return VFADataset(**kwargs)

    if backend in ("lance", "lance_video"):
        from vfa.datasets.vfa_dataset_lance import VFADataset

        kwargs.pop("dataset_metadata", None)  # lance reads its own meta/ sidecar
        if backend == "lance_video":
            kwargs["storage_format"] = "video"
        return VFADataset(**kwargs)

    raise ValueError(f"Unknown dataset backend: {backend!r} (expected 'lerobot', 'lance' or 'lance_video')")
