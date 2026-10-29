"""Isolated, startup-only Torch -> ONNX -> native TensorRT bundle builder."""
import argparse
from importlib import import_module
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .eager import policy_metadata
from .tensorrt import dependencies, file_hash


class Preprocess(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.height, self.width = metadata['processed_hw']
        self.normalize = metadata['normalize']
        self.register_buffer('mean', torch.tensor(metadata['image_mean']).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(metadata['image_std']).view(1, 3, 1, 1))
        self.interpolate = metadata['interpolation']
        self.crop_y, self.crop_x = metadata['crop_offset']

    def forward(self, images):
        b, t, v, _, h, w = images.shape
        x = images.reshape(-1, 3, h, w)
        kwargs = {'align_corners': False} if self.interpolate in ['bilinear', 'bicubic'] else {}
        if self.interpolate:
            x = F.interpolate(x, size=(self.height, self.width), mode=self.interpolate, **kwargs)
        x = x.clamp(0, 1)
        x = x[:, :, self.crop_y:self.crop_y+self.height, self.crop_x:self.crop_x+self.width]
        if self.normalize:
            x = (x - self.mean) / self.std
        return x.reshape(b*t, v, 3, self.height, self.width)


def state_trajectory(policy, normalized):
    tail = torch.zeros((normalized.shape[0], policy.horizon-policy.n_obs_steps, policy.action_dim),
                       device=normalized.device, dtype=normalized.dtype)
    return torch.cat((normalized, tail), dim=1)


class Encoder(nn.Module):
    def __init__(self, policy, metadata, cache):
        super().__init__()
        self.p, self.cache = policy, cache
        self.preprocess = Preprocess(metadata)

    def forward(self, images, state):
        p = self.p
        b = images.shape[0]
        scene = p.obs_encoder.scene_encoder
        tokens = scene.backbone(self.preprocess(images))
        tokens['image_tokens'] = scene.backbone_projection(tokens['image_tokens'])
        features = scene.token_forward(tokens).reshape(b*p.n_image_obs_steps, -1, scene.dim_out)
        features = p.adapt_conditions(features).reshape(b, -1, p.model.hidden_size)
        normalized = p._zero_static_joints(p.normalizer[p.state_key].normalize(state))
        if self.cache:
            kv = p.model.prepare_kv_cache(features)
            kv = torch.stack([torch.stack(pair, dim=1) for pair in kv], dim=1)
            return kv, normalized
        return features, normalized


class Step(nn.Module):
    def __init__(self, policy, cache):
        super().__init__()
        self.p, self.cache = policy, cache

    def forward(self, noisy, normalized, condition, timestep, target_time, dt):
        p = self.p
        trajectory = state_trajectory(p, normalized)
        value = p._zero_static_joints(torch.where(p.action_mask.to(noisy.device), noisy, trajectory))
        if self.cache:
            kv = tuple((condition[:, i, 0], condition[:, i, 1]) for i in range(len(p.model.blocks)))
            scene = None
        else:
            kv, scene = None, condition
        velocity = p.model(p.state_adaptor(value), timestep, scene, kv_cache=kv,
                           r=target_time if p.flow_type == 'meanflow' else None)
        value = p._zero_static_joints(value + dt * p._zero_static_joints(velocity))
        if p.flow_type == 'meanflow':
            value = p._zero_static_joints(torch.where(p.action_mask.to(noisy.device), value, trajectory))
        return value


class Post(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.p = policy

    def forward(self, noisy, normalized, base_state):
        p = self.p
        value = torch.where(p.action_mask.to(noisy.device), noisy, state_trajectory(p, normalized))
        full = p.normalizer[p.supervision_key].unnormalize(value)
        future = full[:, p.n_obs_steps:p.n_obs_steps+p.n_action_steps]
        if 'delta' in p.state_key:
            delta = p._zero_static_joints(future)
            return delta + base_state, full, delta
        return p._hold_static_joints(future, base_state), full


def native_adaptive_pool(x, output_size, use_max_pool=False):
    """Exact regular pooling when adaptive bins are uniform, exact slices otherwise."""
    h, w = map(int, x.shape[-2:])
    oh, ow = map(int, output_size)
    def ranges(n, out):
        return [(math.floor(i*n/out), math.ceil((i+1)*n/out)) for i in range(out)]
    hs, ws = ranges(h, oh), ranges(w, ow)
    def regular(bins):
        kernels = {end-start for start, end in bins}
        strides = {bins[i+1][0]-bins[i][0] for i in range(len(bins)-1)}
        return len(kernels) == 1 and (len(bins) == 1 or len(strides) == 1 and next(iter(strides)) > 0)
    if regular(hs) and regular(ws):
        kernel = (hs[0][1]-hs[0][0], ws[0][1]-ws[0][0])
        stride = (hs[1][0] if oh > 1 else 1, ws[1][0] if ow > 1 else 1)
        fn = F.max_pool2d if use_max_pool else F.avg_pool2d
        return fn(x, kernel_size=kernel, stride=stride)
    rows = []
    for a, b in hs:
        cols = []
        for c, d in ws:
            cell = x[:, :, a:b, c:d]
            cell = cell.amax((-2, -1), keepdim=True) if use_max_pool else cell.mean((-2, -1), keepdim=True)
            cols.append(cell)
        rows.append(torch.cat(cols, dim=-1))
    return torch.cat(rows, dim=-2)


def patch_export():
    ox = import_module('100hzs1.trainer.onnx')
    ox._patch_mha_fastpath_for_onnx()
    ox._patch_torch_onnx_scope_attributes()
    ox._patch_position_getters_for_onnx()
    ox._patch_dinov3_encoders_for_onnx()
    ox._patch_rms_norm_for_onnx()
    adapter = import_module('100hzs1.models.policy.flow_matching.emvis.adapter')
    def forward(self, x, shape_out):
        x = native_adaptive_pool(x, shape_out, self.adapter is nn.AdaptiveMaxPool2d)
        return self.drop(self.proj(x.permute(0, 2, 3, 1)))
    adapter.AdaptivePatchMerging.forward = forward


def metadata_for_export(policy, cache):
    metadata = policy_metadata(policy)
    cfg = policy.config.obs_encoder['emvis_config']
    processor = cfg.get('processor_config', {})
    if any(processor.get(k) for k in ['do_resize', 'do_center_crop', 'do_rescale']):
        raise ValueError('TensorRT export needs processor do_resize/do_center_crop/do_rescale=false; select eager for this custom processor')
    shapes = list(metadata['raw_image_shapes'].values())
    if any(shape != shapes[0] for shape in shapes):
        raise ValueError('TensorRT export requires equal camera resolutions')
    _, h, w = shapes[0]
    patch = int(policy.obs_encoder.scene_encoder.backbone.patch_size_log)
    interpolate = cfg.get('interpolate', 'bilinear')
    if isinstance(interpolate, int):
        mode, target = 'bilinear', interpolate
    elif isinstance(interpolate, (list, tuple)):
        mode, target = interpolate[0], interpolate[1] or w
    else:
        mode, target = interpolate, w
    if mode is False:
        mode = None
    if mode not in [None, 'bilinear', 'bicubic', 'nearest']:
        raise ValueError(f'Unsupported TensorRT interpolation: {mode}')
    if mode:
        tw = round(target / patch) * patch
        th = round(h * (tw / w) / patch) * patch
    else:
        th, tw = min(h//patch, w//patch)*patch, (w//patch)*patch
    if th <= 0 or tw <= 0:
        raise ValueError('Image is smaller than the backbone patch size')
    scale = import_module('100hzs1.models.policy.flow_matching.meanflow').time_embedding_scale
    metadata.update(processed_hw=[th, tw], interpolation=mode,
                    crop_offset=[0, 0] if mode else [(h-th)//2, (w-tw)//2],
                    normalize=bool(processor.get('do_normalize', True)),
                    image_mean=processor.get('image_mean', [.485, .456, .406]),
                    image_std=processor.get('image_std', [.229, .224, .225]),
                    num_timestep_buckets=policy.num_timestep_buckets,
                    time_scale=scale(policy.meanflow_options, policy.num_timestep_buckets) if policy.flow_type == 'meanflow' else float(policy.num_timestep_buckets),
                    static_joint_mask=policy.static_joint_mask.cpu().tolist(), use_kv_cache=cache)
    return metadata


def export_and_build(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    loader = import_module('100hzs1.models.base_runner').load_policy
    p = loader(str(args.checkpoint), device='cpu').float().eval()
    for module in p.modules():
        if module is not p and hasattr(module, 'dtype'):
            try:
                module.dtype = torch.float32
            except AttributeError:
                pass
    p.to(args.device)
    metadata = metadata_for_export(p, args.kv_cache)
    enc, step, post = Encoder(p, metadata, args.kv_cache).to(args.device).eval(), Step(p, args.kv_cache).eval(), Post(p).eval()
    b, h, d = args.batch_size, p.horizon, p.action_dim
    shape = next(iter(metadata['raw_image_shapes'].values()))
    images = torch.rand((b, p.n_image_obs_steps, len(p.obs_encoder.rgb_key), *shape), device=args.device)
    state = torch.zeros((b, p.n_obs_steps, d), device=args.device)
    noisy = torch.zeros((b, h, d), device=args.device)
    time = torch.zeros((b,), device=args.device)
    dt = torch.full((b, 1, 1), .1, device=args.device)
    base = torch.zeros((b, 1, d), device=args.device)
    with torch.no_grad():
        # Ensure our vectorized preprocessing matches the trained image pipeline.
        original = p.obs_encoder.scene_encoder.img_processing(images.reshape(-1, *shape))
        processed = enc.preprocess(images).reshape(-1, *original.shape[-3:])
        torch.testing.assert_close(processed, original, rtol=2e-5, atol=2e-5)
        condition, normalized = enc(images, state)
    patch_export()
    output_names = ['action', 'predicted_raw'] + (['delta_action'] if 'delta' in p.state_key else [])
    condition_name = 'kv' if args.kv_cache else 'scene'
    specs = [
        ('encoder', enc, [images, state], ['images', 'state'], [condition_name, 'normalized']),
        ('step', step, [noisy, normalized, condition, time, time.clone(), dt],
         ['noisy', 'normalized', 'condition', 'timestep', 'target_time', 'dt'], ['next_noisy']),
        ('post', post, [noisy, normalized, base], ['noisy', 'normalized', 'base_state'], output_names),
    ]
    trt, _ = dependencies()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    for name, module, values, inputs, outputs in specs:
        path = args.output / f'{name}.onnx'
        with torch.no_grad():
            torch.onnx.export(module, tuple(values), str(path), opset_version=19, dynamo=False,
                input_names=inputs, output_names=outputs, do_constant_folding=True)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(path)):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError('\n'.join(errors))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.builder_optimization_level = 3
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if args.precision == 'fp16' and name != 'post':
            config.set_flag(trt.BuilderFlag.FP16)
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f'TensorRT {name} build failed')
        (args.output / f'{name}.plan').write_bytes(bytes(plan))
        (args.output / f'{name}.export.json').write_text(json.dumps(dict(
            onnx_sha256=file_hash(path), precision='fp32' if name == 'post' else args.precision,
            batch_size=args.batch_size, builder_optimization_level=3, tf32=False), indent=2))
        # Native inference needs only the plans and metadata, not another copy
        # of the checkpoint's large float weights in ONNX form.
        path.unlink()
        del parser, network, config, plan
    metadata['precision'] = args.precision
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('TENSORRT_BUNDLE_BUILT', args.output, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--kv-cache', action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    export_and_build(args)


if __name__ == '__main__':
    main()
