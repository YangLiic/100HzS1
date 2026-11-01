"""Independent width/compression recipes and true shared-stage4 contracts."""
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
vision = import_module('100hzs1.models.policy.flow_matching.emvis.embodied_vision')
encoder = import_module('100hzs1.models.policy.flow_matching.emvis.encoder.efficientvit_encoder')
adapter = import_module('100hzs1.models.policy.flow_matching.emvis.adapter')
POLICY = 'backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_'
NAMES = {'A': POLICY+'native_fm_d2048_baseline_3v_8gpu_L4',
         'C': POLICY+'stage4repeat_2x2_fm_d2048_3v_8gpu_L4',
         'R': POLICY+'resmlp_fm_d1024_3v_8gpu_L4'}


def recipe(arm):
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'configs/100hzs1_train')):
        cfg=OmegaConf.to_container(compose(config_name=NAMES['C' if arm=='B' else arm]), resolve=True)
    if arm=='B':
        # Legacy six-token behavior is a unit fixture, not a maintained experiment YAML.
        cfg['model']['obs_encoder']['emvis_config']['backbone_config']['stage4_pool_shape']=None
        cfg['model']['obs_encoder']['emvis_config']['model_adapter_config']['shape_out']=[2,3]
        cfg['output_dir'] += '_legacy_fixture'
    return cfg


def scene(arm):
    cfg = deepcopy(recipe(arm)['model']['obs_encoder']['emvis_config'])
    cfg['load_backbone_pretrain'] = False
    return vision.EmVisRM(**cfg)


class VisualAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_three_recipes_preserve_common_training_recipe_and_isolate_model_changes(self):
        configs={key:recipe(key) for key in ['A','C','R']};base=configs['A']
        files={str(p.relative_to(ROOT/'configs/100hzs1_train')).removesuffix('.yaml')
               for p in (ROOT/'configs/100hzs1_train/backbones/efficientvit_b0').rglob('*.yaml')}
        self.assertEqual(files,set(NAMES.values()))
        for arm,cfg in configs.items():
            for key in ['train_dataset','seed','norm','num_train_epochs','train_batch_size','optimizer',
                        'horizon','n_obs_steps','n_image_obs_steps','inference']:
                self.assertEqual(cfg[key],base[key],(arm,key))
            m=cfg['model'];v=m['obs_encoder']['emvis_config']
            width=1024 if arm=='R' else 2048
            self.assertEqual((m['hidden_size'],m['depth'],m['num_heads']),(width,8,width//64))
            self.assertEqual(m['state_adaptor'],'mlp3x_gelu')
            self.assertEqual(m['time_injection'],'tau_token');self.assertEqual(m['flow_type'],'fm')
            self.assertNotIn('ffn_hidden_size',m)
            self.assertEqual(m['scene_adaptor'],'residual_mlp2x_gelu' if arm=='R' else 'mlp2x_gelu')
            self.assertEqual(m['scene_adaptor_hidden_size'],2048 if arm=='R' else None)
            self.assertEqual(v['model_adapter_config']['dim_out'],1024)
            self.assertEqual(v['model_adapter_config']['shape_out'],[2,2])
            self.assertEqual(v['backbone_config']['stage4_repeats'],2 if arm=='C' else 0)
            self.assertEqual(v['backbone_config']['stage4_pool_shape'],[2,2] if arm=='C' else None)
            self.assertFalse(v['backbone_trainable']);self.assertFalse(v['compress_before_fusion'])
            self.assertEqual(v['output_projection_type'],'linear')
            self.assertEqual(cfg['inference']['num_inference_timesteps'],4)
            self.assertEqual(m['noise_scheduler']['num_train_timesteps'],1000)
            self.assertEqual(cfg['norm'],'minmax');self.assertEqual(cfg['seed'],42)
            self.assertEqual(cfg['num_train_epochs'],200);self.assertEqual(cfg['train_batch_size'],128)
            self.assertIsNone(cfg['resume_from_checkpoint'])
        def differences(a,b,path=''):
            if isinstance(a,dict) and isinstance(b,dict):
                return set().union(*(differences(a.get(k),b.get(k),path+'.'+k if path else k) for k in a.keys()|b.keys()))
            return set() if a==b else {path}
        self.assertEqual(differences(base['model'],configs['C']['model']),{
            'obs_encoder.emvis_config.backbone_config.stage4_repeats',
            'obs_encoder.emvis_config.backbone_config.stage4_pool_shape',
            'obs_encoder.emvis_config.model_adapter_config.target',
            'obs_encoder.emvis_config.model_adapter_config.conv_num'})
        self.assertEqual(differences(base['model'],configs['R']['model']),{
            'hidden_size','num_heads','scene_adaptor','scene_adaptor_hidden_size'})
        self.assertEqual(len({cfg['output_dir'] for cfg in configs.values()}),3)

    def test_shared_stage4_changes_grid_and_trains_only_bridge(self):
        m = encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k', stage4_repeats=2).train()
        calls = []
        handle = m.model.stages[-1].register_forward_hook(
            lambda module,args,out: calls.append((tuple(args[0].shape),tuple(out.shape))))
        bn = next(x for x in m.model.modules() if isinstance(x, nn.BatchNorm2d))
        mean = bn.running_mean.clone()
        try:
            output = m(torch.rand(1, 3, 3, 256, 320))
            self.assertEqual(output['image_tokens'].shape, (1,3,6,128))
            self.assertEqual(output['image_tokens_pos'][0,0,-1].tolist(), [1,2])
            self.assertEqual(calls, [((3,64,16,20),(3,128,8,10)),
                                     ((3,64,8,10),(3,128,4,5)),
                                     ((3,64,4,5),(3,128,2,3))])
            output['image_tokens'].square().mean().backward()
            grad = m.stage4_repeat_projection.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.model.parameters()))
            self.assertFalse(bn.training)
            torch.testing.assert_close(mean, bn.running_mean, rtol=0, atol=0)
            names = [name for name,_ in m.named_parameters()]
            self.assertEqual([n for n in names if not n.startswith('model.')],
                             ['stage4_repeat_projection.weight'])
        finally:
            handle.remove()

    def test_fusion_receives_original_grid_for_a_and_compressed_grid_for_b(self):
        for arm, fusion_tokens, output_tokens, width in [('A',80,4,1024),('B',6,6,1024)]:
            with self.subTest(arm=arm):
                m = scene(arm).eval();seen=[]
                handle=m.mv_fuser.register_forward_pre_hook(lambda module,args:seen.append(tuple(args[0].shape)))
                try:
                    with torch.no_grad():out=m(torch.rand(1,3,3,240,320))
                finally:
                    handle.remove()
                self.assertEqual(seen,[(1,3,fusion_tokens,128)])
                self.assertEqual(out.shape,(1,3,output_tokens,width))
                if arm=='B':
                    self.assertIsInstance(m.model_adapter,adapter.TokenProjection)
                    self.assertEqual((m.model_adapter.proj.in_features,m.model_adapter.proj.out_features),(128,1024))
                    self.assertFalse(any(isinstance(x,(adapter.ResBlock,adapter.AdaptivePatchMerging))
                                         for x in m.model_adapter.modules()))

    def test_new_visual_config_strict_json_reload_preserves_shared_repeat_output(self):
        cfg=deepcopy(recipe('B')['model']['obs_encoder']['emvis_config'])
        cfg['load_backbone_pretrain']=False
        first=vision.VisualEncoder(emvis_config=cfg,rgb_key=['left_cam','right_cam','head_cam']).eval()
        restored=vision.VisualEncoder(emvis_config=json.loads(json.dumps(cfg)),rgb_key=first.rgb_key).eval()
        restored.load_state_dict(first.state_dict(),strict=True)
        self.assertEqual(restored.output_shape_meta(),torch.Size([3,2,3,1024]))
        obs={key:torch.rand(1,3,240,320) for key in first.rgb_key}
        with torch.no_grad():torch.testing.assert_close(restored(obs),first(obs),rtol=0,atol=0)

    def test_pool_follows_shared_repeats_before_fusion_and_adds_no_parameters(self):
        legacy=scene('B');m=scene('C').train();events=[];pool_io=[]
        m.load_state_dict(legacy.state_dict(),strict=True)
        self.assertEqual(set(m.state_dict()),set(legacy.state_dict()))
        self.assertEqual(sum(p.numel() for p in m.parameters()),sum(p.numel() for p in legacy.parameters()))
        self.assertEqual(sum(p.numel() for p in m.backbone.stage4_pool.parameters()),0)
        def record_pool(module,args,out):
            events.append(('pool',tuple(out.shape)))
            pool_io.append((args[0].detach().clone(),out.detach().clone()))
        hooks=[m.backbone.model.stages[-1].register_forward_hook(lambda mod,args,out:events.append(('stage4',tuple(out.shape)))),
               m.backbone.stage4_pool.register_forward_hook(record_pool),
               m.mv_fuser.register_forward_pre_hook(lambda mod,args:events.append(('fusion',tuple(args[0].shape))))]
        try:
            out=m(torch.rand(1,3,3,240,320))
            self.assertEqual(events,[('stage4',(3,128,8,10)),('stage4',(3,128,4,5)),
                                     ('stage4',(3,128,2,3)),('pool',(3,128,2,2)),('fusion',(1,3,4,128))])
            self.assertEqual(out.shape,(1,3,4,1024))
            self.assertIsInstance(m.model_adapter,adapter.TokenProjection)
            self.assertFalse(any(isinstance(mod,adapter.ResBlock) for mod in m.modules()))
            x,y=pool_io[0]
            torch.testing.assert_close(y,nn.functional.adaptive_avg_pool2d(x,(2,2)),rtol=0,atol=0)
            # For width3->2, PyTorch averages overlapping pairs [0,1] and [1,2].
            torch.testing.assert_close(y[...,0],x[...,:2].mean(-1),rtol=0,atol=0)
            torch.testing.assert_close(y[...,1],x[...,1:].mean(-1),rtol=0,atol=0)
            out.square().mean().backward()
            for param in [m.backbone.stage4_repeat_projection.weight,next(m.mv_fuser.parameters()),m.model_adapter.proj.weight]:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(param.grad.abs().sum().item(),0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.backbone.model.parameters()))
        finally:
            for hook in hooks:hook.remove()

    def test_pooled_recipe_json_reload_and_invalid_pool_configuration(self):
        cfg=deepcopy(recipe('C')['model']['obs_encoder']['emvis_config']);cfg['load_backbone_pretrain']=False
        first=vision.VisualEncoder(emvis_config=cfg,rgb_key=['left_cam','right_cam','head_cam']).eval()
        second=vision.VisualEncoder(emvis_config=json.loads(json.dumps(cfg)),rgb_key=first.rgb_key).eval()
        second.load_state_dict(first.state_dict(),strict=True)
        self.assertEqual(second.output_shape_meta(),torch.Size([3,2,2,1024]))
        obs={key:torch.rand(1,3,240,320) for key in first.rgb_key}
        with torch.no_grad():torch.testing.assert_close(second(obs),first(obs),rtol=0,atol=0)
        a,b=recipe('B'),recipe('C')
        for key in ['train_dataset','seed','norm','num_train_epochs','train_batch_size','optimizer','horizon']:
            self.assertEqual(a[key],b[key])
        self.assertEqual(a['model']['hidden_size'],b['model']['hidden_size'])
        self.assertEqual(b['model']['obs_encoder']['emvis_config']['backbone_config']['stage4_pool_shape'],[2,2])
        self.assertEqual(a['model']['obs_encoder']['emvis_config']['model_adapter_config']['shape_out'],[2,3])
        self.assertEqual(b['model']['obs_encoder']['emvis_config']['model_adapter_config']['shape_out'],[2,2])
        self.assertNotEqual(a['output_dir'],b['output_dir'])
        for shape in [[0,2],[2,True],[2],2,'2x2']:
            with self.assertRaisesRegex(ValueError,'positive 2D'):
                encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k',stage4_repeats=2,stage4_pool_shape=shape)
        with self.assertRaisesRegex(ValueError,'requires shared stage4'):
            encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k',stage4_pool_shape=[2,2])

    def test_pooled_policy_fm4_cache_and_strict_reload_use_twelve_conditions(self):
        module=import_module('100hzs1.models.policy.flow_matching.dit_fm_policy')
        cfg=deepcopy(recipe('C')['model'])
        cfg.update(hidden_size=64,depth=2,num_heads=4,horizon=4,n_action_steps=3)
        cfg['obs_encoder']['emvis_config']['load_backbone_pretrain']=False
        p=module.S1DitFlowMatching(module.S1DitFlowMatchingConfig(**cfg)).float().train()
        p.model.dtype=p.model.t_embedder.dtype=torch.float32
        p.normalizer.fit({'state':torch.randn(16,14),'action':torch.randn(16,14)})
        nn.init.normal_(p.model.final_layer.ffn_final.fc2.weight,std=.02)
        obs={k:torch.rand(1,1,3,240,320) for k in p.obs_encoder.rgb_key};obs['state']=torch.rand(1,1,14)
        training_obs={**obs,'state':torch.rand(1,4,14)}
        loss=p.compute_loss({'obs':training_obs,'action':{'action':torch.rand(1,4,14)}})
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertGreater(p.obs_encoder.scene_encoder.backbone.stage4_repeat_projection.weight.grad.abs().sum().item(),0)
        p.eval();noise=torch.randn(1,4,14);times=[]
        hook=p.model.register_forward_pre_hook(lambda m,args:times.append(int(args[1][0])))
        with torch.no_grad():cached=p.predict_action(obs,noise_trajectory=noise,use_kv_cache=True,use_cuda_graph=False)
        hook.remove();self.assertEqual(times,[0,250,500,750])
        with torch.no_grad():plain=p.predict_action(obs,noise_trajectory=noise,use_kv_cache=False,use_cuda_graph=False)
        torch.testing.assert_close(cached['action'],plain['action'],rtol=1e-5,atol=1e-6)
        self.assertEqual(p.model.scene_cond_len,12)
        q=module.S1DitFlowMatching(module.S1DitFlowMatchingConfig(**json.loads(p.config.to_json_string()))).float().eval()
        q.load_state_dict(p.state_dict(),strict=True);q.model.dtype=q.model.t_embedder.dtype=torch.float32
        with torch.no_grad():restored=q.predict_action(obs,noise_trajectory=noise,use_kv_cache=True,use_cuda_graph=False)
        torch.testing.assert_close(restored['action'],cached['action'],rtol=0,atol=0)

    def test_legacy_encoder_has_no_extra_state_and_rejects_bad_repeat_counts(self):
        m=encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k')
        self.assertIsNone(m.stage4_repeat_projection)
        self.assertFalse(any('stage4_repeat' in k for k in m.state_dict()))
        for invalid in [-1,True,1.5]:
            with self.assertRaisesRegex(ValueError,'nonnegative integer'):
                encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k',stage4_repeats=invalid)


if __name__=='__main__':unittest.main()
