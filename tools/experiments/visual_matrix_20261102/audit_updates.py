from pathlib import Path
import json,sys,torch
from safetensors import safe_open
R=Path(sys.argv[1]);P=json.loads((R/'plan.json').read_text());O=Path(P['output']);torch.set_num_threads(1)
target=R/'formal_updates_2000_to_4000.json'
if target.exists():print(target.read_text());sys.exit(0)
prefix=P['backbone_prefix'];rows=[]
with safe_open(P['pretrained_file'],framework='pt',device='cpu') as original:
    for kind in ['main','ema']:
        def file(step):return O/f'checkpoint-{step}'/('ema/model.safetensors' if kind=='ema' else 'model.safetensors')
        with safe_open(file(2000),framework='pt',device='cpu') as a,safe_open(file(4000),framework='pt',device='cpu') as b:
            keys=[k for k in a.keys() if k.startswith(prefix)]
            for saved in [a,b]:
                for key in keys:
                    x=saved.get_tensor(key);torch.testing.assert_close(x,original.get_tensor(key[len(prefix):]).to(x.dtype),rtol=0,atol=0)
            groups={'visual_linear':'obs_encoder.scene_encoder.model_adapter.patch_merging.proj.weight','dit_ffn_fc1':'model.blocks.0.ffn.fc1.weight','dit_ffn_fc2':'model.blocks.0.ffn.fc2.weight'}
            if P['repeat_stage4']:
                groups.update(scene_fc1='scene_adaptor.fc1.weight',scene_fc2='scene_adaptor.fc2.weight',scene_norm='scene_adaptor.norm.weight',bridge='obs_encoder.scene_encoder.backbone.stage4_repeat_projection.weight')
            else:
                groups.update(scene_fc1='scene_adaptor.0.weight',scene_fc2='scene_adaptor.2.weight')
                groups['resblock']=next(k for k in a.keys() if k.startswith('obs_encoder.scene_encoder.model_adapter.downsample_blocks.') and k.endswith('.conv1.weight'))
            groups['fusion']=next(k for k in a.keys() if k.startswith('obs_encoder.scene_encoder.mv_fuser.') and k.endswith('.weight') and a.get_tensor(k).ndim==2)
            deltas={}
            for label,key in groups.items():
                delta=b.get_tensor(key).float()-a.get_tensor(key).float();assert torch.isfinite(delta).all()
                count=int((delta!=0).sum());norm=float(delta.norm())
                if kind=='main' or label!='scene_norm':assert count>0 and norm>0,(kind,label)
                deltas[label]=dict(key=key,changed_elements=count,delta_l2=norm)
            rows.append(dict(weights=kind,parameters=deltas,frozen_pretrained_tensors_exact=len(keys),checkpoints=[2000,4000]))
report=dict(status='verified',case=P['case'],proof=rows);target.write_text(json.dumps(report,indent=2));print(json.dumps(report))
