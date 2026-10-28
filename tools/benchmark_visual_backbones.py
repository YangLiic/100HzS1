"""Matched-shape L4 latency: native trunk, aligned tokens and full VFA tower."""
import argparse
from contextlib import nullcontext
import csv
import gc
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
import timm
import transformers

from vfa.models.policy.vfa.emvis.embodied_vision import VisualEncoder

CONFIGS = {
    "dinov3_convnext_base": "blocks_ranking_rgb_dinov3cnnb_3v_8gpu_L4",
    "efficientvit_b2": "blocks_ranking_rgb_efficientvit_b2_3v_8gpu_L4",
    "efficientvit_b3": "blocks_ranking_rgb_efficientvit_b3_3v_8gpu_L4",
}


def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def measure(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    gpu_ms, wall_ms = [], []
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    for _ in range(repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        before = time.perf_counter()
        begin.record()
        function()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - before) * 1000)
        gpu_ms.append(begin.elapsed_time(end))
    return {
        "cuda_median_ms": statistics.median(gpu_ms),
        "cuda_p95_ms": percentile(gpu_ms, .95),
        "wall_median_ms": statistics.median(wall_ms),
        "wall_p95_ms": percentile(wall_ms, .95),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "incremental_peak_mib": (torch.cuda.max_memory_allocated() - baseline) / 1024 ** 2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=150)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--models", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(
        device=torch.cuda.get_device_name(), torch=torch.__version__,
        cuda=torch.version.cuda, transformers=transformers.__version__, timm=timm.__version__,
        python=platform.python_version(), precision="autocast_bfloat16_fp32_weights",
        image_hw=[240, 320], views=3, image_history=1, compile=False,
        warmup=args.warmup, repeats=args.repeats, seed=42,
        scope="GPU-resident RGB float inputs; excludes data loading, BGR correction, H2D and action head",
        nvidia_smi=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,power.limit,clocks.max.sm", "--format=csv"], text=True),
    )
    records = []
    reference_pixels = {}
    for name in args.models:
        with initialize_config_dir(version_base=None, config_dir=str(root / "configs/vfa_train")):
            config = compose(config_name=CONFIGS[name])
        OmegaConf.resolve(config)
        vision_config = OmegaConf.to_container(config.model.obs_encoder, resolve=True)
        torch.manual_seed(42)
        model = VisualEncoder(**vision_config).cuda().eval()
        scene = model.scene_encoder
        (args.output / (name + "_vision_config.json")).write_text(json.dumps(vision_config, indent=2))
        for batch in args.batch_sizes:
            torch.manual_seed(42)
            images = torch.rand(batch, 3, 3, 240, 320, device="cuda")
            obs = {key: images[:, index] for index, key in enumerate(model.rgb_key)}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                processed = scene.img_processing(images.reshape(-1, 3, 240, 320))
                processed = processed.reshape(batch, 3, *processed.shape[1:])
                sample = scene.backbone(processed)
                native_shape = list(sample["image_tokens"].shape)
                output = model(obs)
                assert list(output.shape) == [batch, 12, 1024], output.shape
                key = str(batch)
                if key in reference_pixels:
                    torch.testing.assert_close(processed.cpu(), reference_pixels[key], rtol=0, atol=0)
                else:
                    reference_pixels[key] = processed.cpu().clone()
                def aligned():
                    result = scene.backbone(processed)
                    return scene.backbone_projection(result["image_tokens"])
                functions = {
                    "backbone_native": lambda: scene.backbone(processed),
                    "backbone_aligned_1024": aligned,
                    "full_visual_encoder": lambda: model(obs),
                }
                for scope, function in functions.items():
                    metrics = measure(function, args.warmup, args.repeats)
                    row = dict(
                        model=name, batch_size=batch, images_per_batch=batch * 3, scope=scope,
                        processed_shape=list(processed.shape), native_shape=native_shape,
                        output_shape=list(output.shape),
                        backbone_params=sum(p.numel() for p in scene.backbone.parameters()),
                        bridge_params=sum(p.numel() for p in scene.backbone_projection.parameters()),
                        total_visual_params=sum(p.numel() for p in model.parameters()), **metrics,
                    )
                    row["images_per_second"] = batch * 3 * 1000 / row["wall_median_ms"]
                    row["observations_per_second"] = batch * 1000 / row["wall_median_ms"]
                    records.append(row)
                    print(json.dumps(row), flush=True)
                    (args.output / "latency.json").write_text(json.dumps(dict(environment=environment, measurements=records), indent=2))
            del sample, output, processed, images, obs
        del scene, model, function, functions
        gc.collect()
        torch.cuda.empty_cache()
    with (args.output / "latency.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
