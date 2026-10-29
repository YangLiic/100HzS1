"""Export matched TensorRT components and real, disjoint PTQ/evaluation data.

PyTorch is used only here, never in the latency runtime. Checkpoints are read-only.
"""
import argparse
import gc
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
load_policy = importlib.import_module('100hzs1.models.base_runner').load_policy
ox = importlib.import_module('100hzs1.trainer.onnx')
DATASET = Path('/DRoboticsAILab/vepfs_20260622/licheng.yang/datasets/robotwin2.0_lerobot_clean_joint14/beat_block_hammer')
CKROOT = Path('/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer')
CHECKPOINTS = {
    'dino': CKROOT / 'dinov3cnnb_3v_obs1_h24_8gpu_L4_20261029_v1',
    'b2': CKROOT / 'efficientvit_b2_native_3v_obs1_h24_8gpu_L4_20261029_v1',
}
KEYS = ['left_cam', 'right_cam', 'head_cam']


class Preprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images):
        b = images.shape[0]
        x = images.reshape(-1, 3, 240, 320)
        x = F.interpolate(x, size=(256, 320), mode='bilinear', align_corners=False)
        x = (x.clamp(0, 1) - self.mean) / self.std
        return x.reshape(b, 3, 3, 256, 320)


class Backbone(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.backbone = policy.obs_encoder.scene_encoder.backbone

    def forward(self, images):
        return self.backbone(images)['image_tokens']


class Encoder(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.p = policy
        self.preprocess = Preprocess()

    def forward(self, images, state):
        p = self.p
        images = self.preprocess(images)
        scene = p.obs_encoder.scene_encoder
        token_dict = scene.backbone(images)
        token_dict['image_tokens'] = scene.backbone_projection(token_dict['image_tokens'])
        features = scene.token_forward(token_dict)
        features = features.reshape(images.shape[0], -1, scene.dim_out)
        features = p.adapt_conditions(features)
        cache = p.model.prepare_kv_cache(features)
        kv = torch.stack([torch.stack(pair, dim=1) for pair in cache], dim=1)
        normalized = p._zero_static_joints(p.normalizer[p.state_key].normalize(state))
        return kv, normalized


class Step(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.p = policy

    def forward(self, noisy, normalized, kv, timestep, dt):
        p = self.p
        value = torch.where(p.action_mask.to(noisy.device), noisy, normalized.expand(-1, p.horizon, -1))
        value = p._zero_static_joints(value)
        cache = tuple((kv[:, i, 0], kv[:, i, 1]) for i in range(len(p.model.blocks)))
        velocity = p.model(p.state_adaptor(value), timestep, None, kv_cache=cache)
        velocity = p._zero_static_joints(velocity)
        return p._zero_static_joints(value + dt * velocity)


class Post(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.p = policy

    def forward(self, noisy, state):
        p = self.p
        actions = p.normalizer[p.supervision_key].unnormalize(noisy)[:, p.n_obs_steps:p.n_obs_steps+p.n_action_steps]
        return p._hold_static_joints(actions, state)


def data(out):
    if (out / 'data_manifest.json').exists():
        return
    spec = importlib.util.spec_from_file_location('offline', ROOT / 'scripts/openloop_eval/evaluate.py')
    offline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(offline)
    info = json.loads((DATASET / 'meta/info.json').read_text())
    manifest = {}
    for split, episodes in [('cal', list(range(8))), ('eval', [48, 49])]:
        all_images, all_states, all_targets, indices = [], [], [], []
        for ep in episodes:
            loc = dict(episode_index=ep, episode_chunk=ep // info['chunks_size'])
            length = pq.read_metadata(DATASET / info['data_path'].format(**loc)).num_rows
            anchors = np.linspace(1, length-23, 16, dtype=int).tolist()
            states, actions, images, windows = offline.load_episode(DATASET, info, ep, KEYS, anchors, 1, 1, 23)
            for anchor, window in zip(anchors, windows):
                obs = offline.build_observation(states, images, window, KEYS, 'cpu', 'bgr')
                all_images.append(np.stack([obs[k][0, 0].numpy() for k in KEYS]))
                all_states.append(obs['state'][0].numpy())
                all_targets.append(actions[anchor:anchor+23])
                indices.append([ep, anchor])
        np.save(out / f'{split}_images.npy', np.stack(all_images))
        np.save(out / f'{split}_state.npy', np.stack(all_states))
        np.save(out / f'{split}_target.npy', np.stack(all_targets))
        np.save(out / f'{split}_noise.npy', np.random.default_rng(42).standard_normal((len(indices),24,14)).astype('float32'))
        manifest[split] = dict(episodes=episodes, indices=indices, count=len(indices))
        print('DATA', split, len(indices), flush=True)
    manifest.update(dataset=str(DATASET), cameras=KEYS, color='source BGR converted once to RGB', seed=42)
    (out / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))


def tensor(arr):
    return torch.from_numpy(np.asarray(arr).copy()).cuda()


def export(module, values, names, output_names, path):
    if path.exists():
        return
    axes = {k: {0: 'batch'} for k in names+output_names}
    print('EXPORT', path, flush=True)
    with torch.no_grad():
        torch.onnx.export(module, tuple(values), str(path), input_names=names,
                          output_names=output_names, opset_version=19, dynamo=False,
                          do_constant_folding=True, dynamic_axes=axes)
    print('EXPORTED', path, path.stat().st_size, flush=True)


def model(name, out):
    dest = out / name
    dest.mkdir(exist_ok=True)
    if (dest / 'EXPORTED').exists():
        return
    checkpoint = CHECKPOINTS[name]
    p = load_policy(str(checkpoint), device='cpu').float().cuda().eval()
    assert p.flow_type == 'fm' and p.horizon == 24
    assert list(p.obs_encoder.rgb_key) == KEYS
    if name == 'b2':
        assert p.obs_encoder.scene_encoder.efficientvit_fusion_mode == 'native'
    for mod in p.modules():
        if hasattr(mod, 'dtype') and not isinstance(mod, type(p)):
            try:
                mod.dtype = torch.float32
            except AttributeError:
                pass
    enc, step, post, back = Encoder(p).cuda().eval(), Step(p).eval(), Post(p).eval(), Backbone(p).eval()
    images = tensor(np.load(out/'eval_images.npy', mmap_mode='r')[:1])
    state = tensor(np.load(out/'eval_state.npy', mmap_mode='r')[:1])
    noise = tensor(np.load(out/'eval_noise.npy', mmap_mode='r')[:1])
    with torch.no_grad():
        # Record equivalence before ONNX-only patching, including pixel normalization.
        original = p.obs_encoder.scene_encoder.img_processing(images.reshape(-1,3,240,320))
        processed = enc.preprocess(images).reshape(-1,3,256,320)
        pixel_error = float((original-processed).abs().max())
        torch.testing.assert_close(original, processed, atol=3e-6, rtol=3e-6)
        obs = {k: images[:,i:i+1] for i,k in enumerate(KEYS)}
        original_cond = p.encode_image_conditions(obs,1)
        kv_original = p.model.prepare_kv_cache(original_cond)
        kv, norm = enc(images,state)
        kv_orig = torch.stack([torch.stack(pair,dim=1) for pair in kv_original],dim=1)
        torch.testing.assert_close(kv, kv_orig, atol=2e-4, rtol=2e-4)
        equivalence = {'preprocess_max_abs':pixel_error,'encoder_kv_max_abs':float((kv-kv_orig).abs().max())}
        for steps in [4,10]:
            p.num_inference_timesteps = steps
            ref,_ = p.conditional_sample(original_cond, norm.expand(-1,24,-1),p.action_mask,noise_trajectory=noise,use_kv_cache=True)
            value=noise
            for i in range(steps):
                t=torch.full((1,),int(i/steps*p.num_timestep_buckets),device='cuda',dtype=torch.int64)
                value=step(value,norm,kv,t,torch.full((1,1,1),1/steps,device='cuda'))
            a,b=post(ref,state),post(value,state)
            torch.testing.assert_close(a,b,atol=2e-4,rtol=2e-4)
            equivalence[f'fm{steps}_action_max_abs']=float((a-b).abs().max())
    # Only export helper patches; the interpolation above is already batch-vectorized.
    ox._patch_mha_fastpath_for_onnx()
    ox._patch_torch_onnx_scope_attributes()
    ox._patch_position_getters_for_onnx()
    ox._patch_dinov3_encoders_for_onnx()
    ox._patch_adapter_for_onnx()
    ox._patch_rms_norm_for_onnx()
    with torch.no_grad():
        kv,norm = enc(images,state)
        export(back,[enc.preprocess(images)],['images'],['tokens'],dest/'backbone.onnx')
        export(enc,[images,state],['images','state'],['kv','normalized'],dest/'encoder.onnx')
        t=torch.zeros(1,device='cuda',dtype=torch.int64)
        dt=torch.full((1,1,1),.1,device='cuda')
        export(step,[noise,norm,kv,t,dt],['noisy','normalized','kv','timestep','dt'],['next_noisy'],dest/'step.onnx')
        export(post,[noise,state],['noisy','state'],['action'],dest/'post.onnx')
        for split in ['cal','eval']:
            imgs=np.load(out/f'{split}_images.npy',mmap_mode='r')
            states=np.load(out/f'{split}_state.npy',mmap_mode='r')
            noises=np.load(out/f'{split}_noise.npy',mmap_mode='r')
            accum={}
            for start in range(0,len(imgs),16):
                x,s,n=tensor(imgs[start:start+16]),tensor(states[start:start+16]),tensor(noises[start:start+16])
                b=x.shape[0]
                pix=enc.preprocess(x)
                kv,norm=enc(x,s)
                batch={'kv':kv,'normalized':norm}
                if not (out/f'{split}_processed.npy').exists():
                    batch['processed']=pix
                if split=='eval':
                    batch['tokens']=back(pix)
                for steps in [4,10]:
                    v=n
                    for i in range(steps):
                        batch[f'noisy_{steps}_{i}']=v
                        t=torch.full((b,),int(i/steps*p.num_timestep_buckets),device='cuda',dtype=torch.int64)
                        v=step(v,norm,kv,t,torch.full((b,1,1),1/steps,device='cuda'))
                    if split=='eval':
                        batch[f'action_{steps}']=post(v,s)
                for k,v in batch.items():
                    accum.setdefault(k,[]).append(v.float().cpu().numpy())
                print('REFERENCE',name,split,start+b,'/',len(imgs),flush=True)
            for k,arrs in accum.items():
                location=out/f'{split}_{k}.npy' if k=='processed' else dest/f'{split}_{k}.npy'
                np.save(location,np.concatenate(arrs))
    manifest=dict(checkpoint=str(checkpoint),strict=True,
                  checkpoint_sha256=hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest(),
                  config_sha256=hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest(),
                  fusion_width=p.obs_encoder.scene_encoder.dim_2d,
                  equivalence=equivalence,torch=torch.__version__,tf32=False,
                  cache_shape=list(kv.shape[1:]),normalized_shape=[1,14])
    (dest/'export.json').write_text(json.dumps(manifest,indent=2))
    (dest/'EXPORTED').write_text('OK\n')
    print('MODEL_EXPORTED',name,flush=True)
    del p,enc,step,post,back
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--model',choices=['dino','b2','all'],default='all')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data(args.output)
    for name in CHECKPOINTS if args.model=='all' else [args.model]:
        model(name,args.output)


if __name__=='__main__':
    main()
