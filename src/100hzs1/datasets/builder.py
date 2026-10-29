"""Dataset backend selector.

`train_dataset._target_` points here; `backend: lerobot|lance|lance_video` picks the
implementation. Both run in the same env (e.g. 100HzS1): the lance backend
(lance_dataset.py) reads via lancedb/pylance directly and does not
import lerobot, so the two never conflict.
"""


def build_dataset(backend: str = "lerobot", **kwargs):
    if backend == "lerobot":
        from .dataset import S1Dataset

        return S1Dataset(**kwargs)

    if backend in ("lance", "lance_video"):
        from .lance_dataset import S1Dataset

        kwargs.pop("dataset_metadata", None)  # lance reads its own meta/ sidecar
        if backend == "lance_video":
            kwargs["storage_format"] = "video"
        return S1Dataset(**kwargs)

    raise ValueError(f"Unknown dataset backend: {backend!r} (expected 'lerobot', 'lance' or 'lance_video')")
