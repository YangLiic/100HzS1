"""Independent trainable MBConv compression after a frozen pretrained trunk."""
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import save_file
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
enc=import_module('100hzs1.models.policy.flow_matching.emvis.encoder.efficientvit_encoder')
vis=import_module('100hzs1.models.policy.flow_matching.emvis.embodied_vision')
pol=import_module('100hzs1.models.policy.flow_matching.dit_fm_policy')
NAME='backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_mbconv2x_2x3_meanflow_d2048_3v_8gpu_L4'

def config():
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs/100hzs1_train')):
        return OmegaConf.to_container(compose(config_name=NAME),resolve=True)

class MBConvDownsamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_recipe_and_incompatible_settings(self):
        cfg=config();m=cfg['model'];v=m['obs_encoder']['emvis_config']
        self.assertEqual((m['hidden_size'],m['num_heads'],m['depth']),(2048,32,8))
        self.assertEqual((m['flow_type'],m['time_injection']),('meanflow','tau_token'))
        self.assertEqual((m['scene_adaptor'],m['state_adaptor']),('mlp2x_gelu','mlp3x_gelu'))
        self.assertEqual(v['backbone_config'],dict(model_name='efficientvit_b0.r224_in1k',stage4_repeats=0,
                                                 mbconv_downsample_blocks=2,mbconv_downsample_init='stage4_local'))
        self.assertFalse(v['compress_before_fusion']);self.assertFalse(v['backbone_trainable'])
        self.assertEqual(v['model_adapter_config']['target'],'TokenProjection')
        self.assertEqual(v['model_adapter_config']['shape_out'],[2,3])
        self.assertEqual(v['model_adapter_config']['dim_out'],1024)
        self.assertEqual((cfg['num_train_epochs'],cfg['train_batch_size'],cfg['horizon']),(200,128,24))
        self.assertTrue(cfg['train_dataset']['bgr_to_rgb'])
        for args in [dict(stage4_repeats=1,mbconv_downsample_blocks=2),dict(mbconv_downsample_blocks=-1),
                     dict(mbconv_downsample_blocks=True),dict(mbconv_downsample_blocks=3),
                     dict(mbconv_downsample_init='invalid')]:
            with self.assertRaises(ValueError):enc.EfficientViTEncoder('efficientvit_b0.r224_in1k',**args)

    def test_explicit_pretrained_load_refreshes_initializers_without_aliases(self):
        m=enc.EfficientViTEncoder('efficientvit_b0.r224_in1k',mbconv_downsample_blocks=2)
        state={k:v.clone() for k,v in m.model.state_dict().items()}
        # Make the file differ from the constructor, exercising the load order.
        for k in state:
            if 'local_module.main.inverted_conv.conv.weight' in k:state[k].add_(.125)
        with tempfile.TemporaryDirectory(prefix='s1_mbconv_init_') as directory:
            path=Path(directory)/'model.safetensors';save_file(state,str(path));m.load_pretrained_model(path,freeze=True)
        for i,block in enumerate(m.mbconv_downsample_layers):
            source=m.model.stages[-1].blocks[i+1].local_module.main
            for key,value in source.state_dict().items():torch.testing.assert_close(block.state_dict()[key],value,rtol=0,atol=0)
            self.assertEqual(block.depth_conv.conv.stride,(2,2))
            self.assertEqual(source.depth_conv.conv.stride,(1,1))
            self.assertTrue(all(a.data_ptr()!=b.data_ptr() for a,b in zip(block.parameters(),source.parameters())))
            self.assertTrue(all(p.requires_grad for p in block.parameters()))
        self.assertTrue(all(a.data_ptr()!=b.data_ptr() for a,b in zip(*[x.parameters() for x in m.mbconv_downsample_layers])))
        self.assertTrue(all(not p.requires_grad for p in m.model.parameters()))

    def test_shapes_single_stage4_call_gradients_and_separate_bn_modes(self):
        m=enc.EfficientViTEncoder('efficientvit_b0.r224_in1k',mbconv_downsample_blocks=2).train()
        seen=[];hook=m.model.stages[-1].register_forward_hook(lambda mod,args,out:seen.append(tuple(out.shape)))
        bns=[x for x in m.model.modules() if isinstance(x,nn.BatchNorm2d)]
        originals=[x.running_mean.clone() for x in bns]
        self.assertTrue(all(not x.training for x in bns))
        self.assertTrue(all(x.training for x in m.mbconv_downsample_layers.modules() if isinstance(x,nn.BatchNorm2d)))
        shapes=[];hooks=[x.register_forward_hook(lambda mod,args,out:shapes.append(tuple(out.shape))) for x in m.mbconv_downsample_layers]
        out=m(torch.rand(1,3,3,256,320));hook.remove()
        for h in hooks:h.remove()
        self.assertEqual(seen,[(3,128,8,10)])
        self.assertEqual(shapes,[(3,128,4,5),(3,128,2,3)])
        self.assertEqual(out['image_tokens'].shape,(1,3,6,128))
        self.assertEqual(out['image_tokens_pos'][0,0,-1].tolist(),[1,2])
        (out['image_tokens']*torch.randn_like(out['image_tokens'])).sum().backward()
        for block in m.mbconv_downsample_layers:
            g=block.inverted_conv.conv.weight.grad
            self.assertIsNotNone(g);self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().sum().item(),0)
            self.assertGreater(block.point_conv.norm.num_batches_tracked.item(),0)
        self.assertTrue(all(p.grad is None for p in m.model.parameters()))
        for bn,old in zip(bns,originals):torch.testing.assert_close(bn.running_mean,old,rtol=0,atol=0)

    def test_fusion_gets_six_tokens_with_no_bridge_resnet_or_attention_copies(self):
        cfg=deepcopy(config()['model']['obs_encoder']['emvis_config']);cfg['load_backbone_pretrain']=False
        m=vis.EmVisRM(**cfg).eval();seen=[]
        hook=m.mv_fuser.register_forward_pre_hook(lambda mod,args:seen.append(tuple(args[0].shape)))
        with torch.no_grad():out=m(torch.rand(1,3,3,240,320))
        hook.remove();self.assertEqual(seen,[(1,3,6,128)]);self.assertEqual(out.shape,(1,3,6,1024))
        self.assertIsNone(m.backbone.stage4_repeat_projection)
        self.assertEqual(m.backbone.stage4_repeats,0)
        self.assertEqual([type(x).__name__ for x in m.backbone.mbconv_downsample_layers],['MBConv','MBConv'])
        self.assertFalse(any(isinstance(x,(nn.AdaptiveAvgPool2d,nn.AdaptiveMaxPool2d)) for x in m.model_adapter.modules()))

    def test_two_meanflow_updates_reach_both_new_blocks(self):
        cfg=deepcopy(config()['model']);cfg['obs_encoder']['emvis_config']['load_backbone_pretrain']=False
        cfg.update(hidden_size=64,num_heads=1,depth=1)
        p=pol.S1DitFlowMatching(pol.S1DitFlowMatchingConfig(**cfg)).float().train()
        p.model.dtype=p.model.t_embedder.dtype=p.model.r_embedder.dtype=torch.float32
        nn.init.normal_(p.model.final_layer.ffn_final.fc2.weight,std=.02)
        p.normalizer.fit({'state':torch.randn(32,14),'action':torch.randn(32,14)})
        optimizer=torch.optim.AdamW([x for x in p.parameters() if x.requires_grad],lr=1e-4)
        obs={k:torch.rand(1,1,3,240,320) for k in p.obs_encoder.rgb_key};obs['state']=torch.rand(1,24,14)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True);result=p.compute_loss({'obs':obs,'action':{'action':torch.rand(1,24,14)}})
            loss=result['loss'] if isinstance(result,dict) else result
            self.assertTrue(torch.isfinite(loss));loss.backward()
            for block in p.obs_encoder.scene_encoder.backbone.mbconv_downsample_layers:
                g=block.inverted_conv.conv.weight.grad
                self.assertIsNotNone(g);self.assertTrue(torch.isfinite(g).all());self.assertGreater(g.abs().sum().item(),0)
            optimizer.step()

    def test_json_strict_reload_does_not_require_pretrained_file(self):
        cfg=deepcopy(config()['model']['obs_encoder']['emvis_config']);cfg['load_backbone_pretrain']=False
        cfg['backbone_path']='/nonexistent/pretrained'
        a=vis.VisualEncoder(emvis_config=cfg,rgb_key=['left_cam','right_cam','head_cam']).eval()
        b=vis.VisualEncoder(emvis_config=json.loads(json.dumps(cfg)),rgb_key=a.rgb_key).eval()
        b.load_state_dict(a.state_dict(),strict=True)
        obs={k:torch.rand(1,3,240,320) for k in a.rgb_key}
        with torch.no_grad():torch.testing.assert_close(a(obs),b(obs),rtol=0,atol=0)
        self.assertEqual(b.output_shape_meta(),torch.Size([3,2,3,1024]))

if __name__=='__main__':unittest.main()
