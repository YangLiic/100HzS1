"""B1 shared-stage4 spatial/gradient contract and overnight recipe isolation."""
import copy, importlib, unittest
from pathlib import Path
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf

E=importlib.import_module('100hzs1.models.policy.flow_matching.emvis.encoder.efficientvit_encoder').EfficientViTEncoder
M=importlib.import_module('100hzs1.models.policy.flow_matching.emvis.embodied_vision').EmVisRM
C=importlib.import_module('100hzs1.models.policy.flow_matching.dit_fm_policy').S1DitFlowMatchingConfig
ROOT=Path(__file__).resolve().parents[1]
class B1SharedStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_shared_stage_shapes_gradients_and_frozen_buffers(self):
        torch.manual_seed(42);m=E('efficientvit_b1.r256_in1k',visual_variant='stage4_rms').train();calls=[]
        before={k:v.clone() for k,v in m.model.state_dict().items()}
        h=m.model.stages[-1].register_forward_hook(lambda mod,args,out:calls.append((list(args[0].shape),list(out.shape))))
        try:o=m(torch.rand(1,3,3,256,320))['image_tokens']
        finally:h.remove()
        self.assertEqual(list(o.shape),[1,3,6,256])
        self.assertEqual(calls,[([3,128,16,20],[3,256,8,10]),([3,128,8,10],[3,256,4,5]),([3,128,4,5],[3,256,2,3])])
        o.square().mean().backward();self.assertEqual(list(m.stage4_repeat_projection.weight.shape),[128,256,1,1])
        self.assertGreater(m.stage4_repeat_projection.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(q.grad is None and not q.requires_grad for q in m.model.parameters()))
        for k,v in m.model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        restored=E('efficientvit_b1.r256_in1k',visual_variant='stage4_rms').eval();restored.load_state_dict(m.state_dict(),strict=True)
    def test_all_32_recipes_and_b1_fusion_pool_order(self):
        OmegaConf.register_new_resolver('eval',eval,replace=True)
        root=ROOT/'configs/100hzs1_train/experiments/visual_matrix_20261102'
        files=sorted(p for p in root.glob('*.yaml') if p.stem!='base');self.assertEqual(len(files),32)
        for f in files:
            with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs/100hzs1_train')):
                cfg=OmegaConf.to_container(compose(config_name='experiments/visual_matrix_20261102/'+f.stem),resolve=True)
            with self.subTest(case=f.stem):
                p=C(**copy.deepcopy(cfg['model']));self.assertEqual(p.flow_type,'fm');self.assertEqual(p.time_injection,'tau_token')
                self.assertEqual(cfg['norm'],'groot');self.assertEqual(cfg['inference']['precision'],'bf16')
                self.assertEqual(cfg['train_dataset']['task'],cfg['experiment_task']);self.assertEqual(cfg['train_dataset']['demo'],50)
                self.assertIn(cfg['experiment_case'],cfg['output_dir']);self.assertEqual(cfg['num_train_epochs'],200)
                if f.stem.startswith('b1_adjust_bottle_stage4_rms'):
                    v=copy.deepcopy(cfg['model']['obs_encoder']['emvis_config']);v['load_backbone_pretrain']=False
                    m=M(**v).train();events=[];h=m.mv_fuser.register_forward_pre_hook(lambda mod,args:events.append(list(args[0].shape)))
                    try:o=m(torch.rand(1,3,3,240,320))
                    finally:h.remove()
                    self.assertEqual(events,[[1,3,6,256]]);self.assertEqual(list(o.shape),[1,3,4 if cfg['experiment_pooling'] else 6,1024])
                    o.square().mean().backward();self.assertGreater(m.backbone.stage4_repeat_projection.weight.grad.abs().sum().item(),0)
if __name__=='__main__':unittest.main()
