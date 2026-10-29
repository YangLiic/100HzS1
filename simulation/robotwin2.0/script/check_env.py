"""Validate the unified environment and optionally perform a real GPU render."""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parents[1] / "src"), str(ROOT)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch

    for name in [
        "sapien",
        "mplib",
        "open3d",
        "cv2",
        "100hzs1",
        "policy.100hzs1",
        "envs.beat_block_hammer",
        "curobo.wrap.reacher.motion_gen",
    ]:
        importlib.import_module(name)
    assert torch.__version__ == "2.7.1+cu126", torch.__version__
    assert np.__version__ == "1.26.4", np.__version__
    assert Path(torch.__file__).is_relative_to(Path(sys.prefix))
    leaked = [
        str(getattr(m, "__file__", ""))
        for m in sys.modules.values()
        if "DRRM-VODPP" in str(getattr(m, "__file__", ""))
    ]
    assert not leaked, leaked
    result = {
        "python": sys.executable,
        "environment": sys.prefix,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sapien": importlib.metadata.version("sapien"),
        "cuda_available": torch.cuda.is_available(),
        "render": False,
    }
    if args.render:
        assert torch.cuda.is_available(), "An available NVIDIA GPU is required"
        import sapien.core as sapien

        engine = sapien.Engine()
        engine.set_renderer(sapien.SapienRenderer())
        scene = sapien.Scene([sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem("cuda:0")])
        scene.add_ground(0)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        camera = scene.add_camera(
            "check", width=320, height=240, fovy=1.0, near=0.1, far=10
        )
        camera.set_pose(sapien.Pose([0, 0, 1], [1, 0, 0, 0]))
        scene.step()
        scene.update_render()
        camera.take_picture()
        pixels = camera.get_picture("Color")
        assert pixels.shape == (240, 320, 4) and np.isfinite(pixels).all(), pixels.shape
        result.update(
            render=True,
            gpu=torch.cuda.get_device_name(0),
            rgba_shape=list(pixels.shape),
            render_cuda_id=scene.render_system.device.cuda_id,
            render_pci=scene.render_system.device.pci_string,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
