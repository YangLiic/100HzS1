from pathlib import Path
import hashlib, torch
from safetensors import safe_open

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_model(p,P,obs,weight_file=None):
    D=P['hidden_size'];N=P['num_conditions'];pool=P['use_pooling'];stage=P['repeat_stage4'];C=P['backbone_channels'];V=1024
    assert p.config.norm=='groot' and p.config.visual_variant==P['variant']
    assert p.config.hidden_size==D and p.config.visual_pooling is pool
    assert p.config.num_heads==D//64 and p.config.depth==8
    assert p.flow_type=='fm' and p.meanflow_options is None and p.model.r_embedder is None
    assert p.model.time_injection=='tau_token' and p.num_inference_timesteps==4 and p.num_timestep_buckets==1000
    for field in ['state','action']:
        n=p.normalizer[field];stats=n.params_dict['input_stats'];assert 'q01' in stats and 'q99' in stats and 'clip_bounds' in n.params_dict
        extremes=torch.stack([torch.full_like(stats['q01'],-1e6),torch.full_like(stats['q01'],1e6)])
        assert torch.isfinite(n.normalize(extremes)).all() and n.normalize(extremes).abs().max()<=1
    s=p.obs_encoder.scene_encoder;b=s.backbone;a=s.model_adapter
    assert list(p.obs_encoder.output_shape_meta())==[3,*P['grid'],V]
    assert getattr(b,'shared_stage4',False)==stage
    assert len(a.downsample_blocks)==(0 if stage else 2)
    assert a.patch_merging.use_pooling is pool
    assert (a.patch_merging.proj.in_features,a.patch_merging.proj.out_features)==(C,V)
    trunk=b.model if P['backbone']!='dinol' else b
    assert not any(q.requires_grad for q in trunk.parameters())
    assert all(not m.training for m in trunk.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm))
    if stage:
        bridge=b.stage4_repeat_projection.weight
        assert bridge.requires_grad and list(bridge.shape)==[C//2,C,1,1]
    frozen_count=0
    with safe_open(P['pretrained_file'],framework='pt',device='cpu') as original:
        saved=safe_open(weight_file,framework='pt',device='cpu') if weight_file else None
        try:
            for key,value in trunk.state_dict().items():
                dtype=saved.get_tensor(P['backbone_prefix']+key).dtype if saved else value.dtype
                torch.testing.assert_close(value,original.get_tensor(key).to(dtype).to(value.dtype),rtol=0,atol=0)
                frozen_count+=1
        finally:
            if saved is not None:del saved
    def linears(module):return [[m.in_features,m.out_features] for m in module.modules() if isinstance(m,torch.nn.Linear)]
    assert linears(p.scene_adaptor)==([[1024,2048],[2048,1024]] if stage else [[1024,2048],[2048,2048]])
    if stage:
        assert type(p.scene_adaptor.norm).__name__=='RmsNorm'
        x=torch.randn(1,N,D);m=p.scene_adaptor
        with torch.no_grad():torch.testing.assert_close(m(x),x+m.fc2(m.act(m.fc1(m.norm(x)))),rtol=0,atol=0)
    else:assert [type(m) for m in p.scene_adaptor]==[torch.nn.Linear,torch.nn.GELU,torch.nn.Linear]
    assert linears(p.state_adaptor)==[[14,D],[D,D],[D,D]]
    assert p.model.t_embedder.mlp[-1].out_features==D
    assert [[x.ffn.fc1.in_features,x.ffn.fc1.out_features,x.ffn.fc2.in_features,x.ffn.fc2.out_features] for x in p.model.blocks]==[[D,D,D,D]]*8
    assert sum(isinstance(m,torch.nn.LayerNorm) for m in s.mv_fuser.modules())==5
    events=[];outputs=[]
    hooks=[s.mv_fuser.register_forward_pre_hook(lambda m,args:events.append(['fusion',list(args[0].shape)])),
           a.patch_merging.register_forward_pre_hook(lambda m,args:events.append(['pool_projection',list(args[0].shape)])),
           b.register_forward_hook(lambda m,args,out:outputs.append(list(out['image_tokens'].shape)))]
    stage_calls=[]
    if P['backbone']!='dinol':hooks.append(b.model.stages[-1].register_forward_hook(lambda m,args,out:stage_calls.append(list(out.shape))))
    try:
        with torch.no_grad():conditions=p.encode_image_conditions(obs,1)
    finally:
        for h in hooks:h.remove()
    assert outputs==[[1,3,6 if stage else 80,C]],outputs
    assert events==[['fusion',[1,3,6 if stage else 80,C]],['pool_projection',[3,C,2,3]]],events
    if P['backbone']!='dinol':assert stage_calls==([[3,C,8,10],[3,C,4,5],[3,C,2,3]] if stage else [[3,C,8,10]])
    assert list(conditions.shape)==[1,N,D]
    p.model.dtype=p.model.t_embedder.dtype=torch.float32
    with torch.no_grad():kv=p.model.prepare_kv_cache(conditions)
    assert len(kv)==8 and all(list(k.shape)==list(v.shape)==[1,D//64,N,64] for k,v in kv)
    times=[];h=p.model.register_forward_pre_hook(lambda m,args:times.append(int(args[1][0])))
    try:
        with torch.no_grad():out=p.predict_action(obs,use_cuda_graph=False,use_kv_cache=True)
    finally:h.remove()
    assert times==[0,250,500,750] and list(out['action'].shape)==[1,23,14]
    assert torch.isfinite(out['action']).all()
    return dict(status='passed',hidden_size=D,conditions_shape=list(conditions.shape),kv_shape=list(kv[0][0].shape),
                actual_sampling_time_buckets=times,visual_linear=[C,1024],scene_mlp=linears(p.scene_adaptor),
                forward_events=events,stage4_call_output_shapes=stage_calls,pooling_enabled=pool,pool_after_fusion=True,
                frozen_pretrained_tensors_exact=frozen_count,original_trunk_frozen=True,action_shape=list(out['action'].shape))
