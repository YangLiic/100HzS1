from pathlib import Path
from importlib import import_module
import copy,json,sys,os,torch,hydra
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from checks import verify_model,sha
R=Path(sys.argv[1]);P=json.loads((R/'plan.json').read_text());S=Path(P['snapshot']);os.chdir(S)
torch.set_num_threads(2);torch.manual_seed(42)
OmegaConf.register_new_resolver('eval',eval,replace=True)
with initialize_config_dir(version_base=None,config_dir=str(S/'configs/100hzs1_train')):
    cfg=compose(config_name=P['config_name'])
raw=OmegaConf.to_container(cfg,resolve=True);d=raw['train_dataset'];m=raw['model']
assert raw['output_dir']==P['output'] and d['task']==P['task'] and d['demo']==50
assert raw['norm']=='groot' and raw['seed']==42 and raw['mixed_precision']=='bf16'
assert raw['num_train_epochs']==200 and raw['train_batch_size']==128 and raw['horizon']==24
assert raw['resume_from_checkpoint'] is None and not raw.get('checkpoint_dir')
assert d['bgr_to_rgb'] and d['cache_to_ram'] and d['cache_decoded_video']
assert m['inference']==raw['inference']
for key,val in dict(backend='eager',precision='bf16',num_inference_timesteps=4,use_kv_cache=True,use_cuda_graph=False).items():assert raw['inference'][key]==val
Path(P['training_config']).write_text(OmegaConf.to_yaml(cfg,resolve=True))
cls=import_module('100hzs1.models.policy.flow_matching.dit_fm_policy')
p=cls.S1DitFlowMatching(cls.S1DitFlowMatchingConfig(**copy.deepcopy(m))).float().eval()
probe=OmegaConf.create(d);probe.cache_to_ram=False;probe.cache_decoded_video=False
dataset=hydra.utils.instantiate(probe);val=dataset.get_validation_dataset();sample=dataset[0]
p.set_normalizer(dataset.get_normalizer(norm='groot'));obs={k:v.unsqueeze(0) for k,v in sample['obs'].items()}
report=verify_model(p,P,obs)
p.zero_grad(set_to_none=True);p.encode_image_conditions(obs,1).square().mean().backward()
s=p.obs_encoder.scene_encoder;groups={'visual_linear':s.model_adapter.patch_merging.proj.weight,'fusion':next(s.mv_fuser.parameters())}
for name,q in p.scene_adaptor.named_parameters():
    if name.endswith('weight'):groups['scene.'+name]=q
if P['repeat_stage4']:groups['bridge']=s.backbone.stage4_repeat_projection.weight
grads={}
for key,q in groups.items():
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum()>0,key
    grads[key]=float(q.grad.norm())
steps=len(dataset)*200//128;assert steps>4000
report.update(train_samples=len(dataset),val_samples=len(val),planned_optimizer_steps=steps,visual_gradients=grads,
              total_parameters=sum(q.numel() for q in p.parameters()),trainable_parameters=sum(q.numel() for q in p.parameters() if q.requires_grad),
              optimizer_steps=0,norm='groot',dataset_root=d['root'])
(R/'preflight.json').write_text(json.dumps(report,indent=2))
P.update(training_config_sha256=sha(P['training_config']),planned_optimizer_steps=steps)
(R/'plan.json').write_text(json.dumps(P,indent=2));print(json.dumps(report),flush=True)
