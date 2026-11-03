from pathlib import Path
import ast,copy,hashlib,json,os,subprocess,tarfile,yaml
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
B=Path('/data/vepfs/licheng.yang/100HzS1');NAME='robotwin_visual_matrix_fm4_20261102_v1';R=B/'runs'/NAME
assert not R.exists();R.mkdir();S=R/'snapshot';S.mkdir()
COMMIT=subprocess.check_output(['git','rev-parse','HEAD'],cwd=B,text=True).strip()
with (R/'source.tar').open('xb') as f:subprocess.run(['git','archive',COMMIT],cwd=B,stdout=f,check=True)
with tarfile.open(R/'source.tar') as f:f.extractall(S)
overlay=['src/100hzs1/models/policy/flow_matching/emvis/encoder/efficientvit_encoder.py','src/100hzs1/models/policy/flow_matching/visual_variants.py','tests/test_b1_shared_stage4.py']
overlay += [str(p.relative_to(B)) for root in ['configs/100hzs1_train/experiments/visual_matrix_20261102','tools/experiments/visual_matrix_20261102'] for p in (B/root).rglob('*') if p.is_file()]
for rel in overlay:
    dest=S/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((B/rel).read_bytes())
OLD=B/'runs/beat_b0_groot_nopool_six_fm4_20261102_v1';links=json.loads((OLD/'plan.json').read_text())['runtime_links']
for rel,target in links.items():
    p=S/rel;assert Path(target).exists() and not p.exists();p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(target,target_is_directory=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes={str(p.relative_to(S)):sha(p) for root in ['src','configs','scripts','simulation/robotwin2.0','tools/experiments/visual_matrix_20261102'] for p in (S/root).rglob('*') if p.is_file() and not p.is_symlink()};hashes['main.py']=sha(S/'main.py')
H=B/'tools/experiments/visual_matrix_20261102';ref=OLD/'cases/stage4_rms'
pretrained={'b0':S/'pretrained/efficientvit_b0.r224_in1k/model.safetensors','b1':S/'pretrained/efficientvit_b1.r256_in1k/model.safetensors','dinol':Path('/data/vepfs/licheng.yang/DRRM-VODPP/pretrained/dinov3-cnnl/model.safetensors')}
ph={k:sha(v) for k,v in pretrained.items()}
specs=json.loads((B/'runs/visual_matrix_20261102_specs.json').read_text())
OmegaConf.register_new_resolver('eval',eval,replace=True)
cases=[];nodes=[];groups=[]
for machine,job,bb,dtasks in [('machine1','t-20260820125318-wd2mn','b0',['blocks_ranking_rgb','beat_block_hammer']),('machine2','t-20260819215926-zt9qr','b1',['adjust_bottle','click_alarmclock'])]:
    N=R/'nodes'/machine;N.mkdir(parents=True);request=f's1-visual-matrix-fm4-{machine}-20261102-v1'
    ordered=[(bb,t) for t in ['click_alarmclock','adjust_bottle','blocks_ranking_rgb']]+[('dinol',t) for t in dtasks]
    node_groups=[]
    for idx,(backbone,task) in enumerate(ordered):
        G=N/'groups'/f'{idx:02d}_{backbone}_{task}';G.mkdir(parents=True)
        for name in ['queue/pending','queue/claimed','queue/done','queue/failed','logs']:(G/name).mkdir(parents=True)
        roots=[]
        for spec in [x for x in specs if x['backbone']==backbone and x['task']==task]:
            name=spec['case'];C=R/'cases'/name;C.mkdir(parents=True);(C/'configs').mkdir();E=C/'eval';(E/'seeds').mkdir(parents=True)
            O=f'/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/{task}/{NAME}/{name}'
            T=f'/DRoboticsAILab/users/licheng.yang/100HzS1_tmp/{NAME}/{name}'
            EO=f'/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/{task}/{NAME}/{name}/main_nfe4_100_v1'
            assert not any(Path(x).exists() for x in [O,T,EO])
            config_name='experiments/visual_matrix_20261102/'+name
            os.environ['S1_ROOT']=str(S)
            with initialize_config_dir(version_base=None,config_dir=str(S/'configs/100hzs1_train')):cfg=compose(config_name=config_name)
            raw=OmegaConf.to_container(cfg,resolve=True);assert raw['output_dir']==O
            cf=C/'configs/train.yaml';cf.write_text(OmegaConf.to_yaml(cfg,resolve=True))
            stage=spec['variant']=='stage4_rms';D=1024 if stage else 2048;channels={'b0':128,'b1':256,'dinol':1536}[backbone]
            P=dict(name=NAME+'_'+name,campaign=NAME,case=name,variant=spec['variant'],task=task,backbone=backbone,
                machine=machine,job_id=job,request_id=request,commit=COMMIT,run_root=str(C),snapshot=str(S),output=O,smoke_output=T,eval_output=EO,eval_root=str(E),
                hidden_size=D,backbone_channels=channels,visual_width=1024,repeat_stage4=stage,scene_kind='rms' if stage else 'plain',use_pooling=spec['pooling'],
                num_conditions=12 if spec['pooling'] else 18,grid=[2,2] if spec['pooling'] else [2,3],norm='groot',
                backbone_prefix='obs_encoder.scene_encoder.backbone.'+('' if backbone=='dinol' else 'model.'),
                flow_type='fm',nfe=4,formal_training_runs=1,smoke_optimizer_steps=0,planned_optimizer_steps=None,
                training_config=str(cf),training_config_sha256=sha(cf),source_hashes=hashes,runtime_links=links,config_name=config_name,config_overrides=[],
                pretrained_file=str(pretrained[backbone]),pretrained_sha256=ph[backbone],eval_tmpdir=f'/data/vepfs/licheng.yang/tmp/{NAME}_{name}',seed_list=str(E/'seeds/all.txt'))
            seed=S/f'simulation/robotwin2.0/task_config/seedlists/{task}_demo_clean.txt';seeds=list(map(int,seed.read_text().split()));meta=json.loads(seed.with_suffix('.expert.json').read_text())
            assert len(seeds)==len(set(seeds))==100 and meta['verified'] and meta['task']==task and meta['seed_sha256']==sha(seed)
            (E/'seeds/all.txt').write_bytes(seed.read_bytes());(E/'seeds/all.expert.json').write_bytes(seed.with_suffix('.expert.json').read_bytes());shards=[]
            for i in range(4):
                f=E/f'seeds/shard{i}.txt';f.write_text(''.join(str(x)+'\n' for x in seeds[i*25:(i+1)*25]))
                f.with_suffix('.expert.json').write_text(json.dumps(dict(verified=True,task=task,task_config='demo_clean',count=25,seed_sha256=sha(f),selection='Contiguous subset of committed expert-verified 100 seeds',parent_seed_sha256=sha(seed),parent_metadata=str(seed.with_suffix('.expert.json'))),indent=2))
                item=dict(id=f'shard{i}',case=name,case_plan=str(C/'plan.json'),episodes=25,seed_list=str(f),output=EO+f'/shard{i}');shards.append(item)
                (G/f'queue/pending/{name}_{i}.json').write_text(json.dumps(item,indent=2))
            P['shards']=shards;P['seed_hashes']={str(f.relative_to(E)):sha(f) for f in (E/'seeds').iterdir()};(C/'plan.json').write_text(json.dumps(P,indent=2))
            for filename in ['checks.py','preflight.py','audit_updates.py']:(C/filename).write_bytes((H/filename).read_bytes())
            for filename in ['launch.py','wandb_relay.py','strict_reload.py','smoke_ddp.py']:
                text=(ref/filename).read_text()
                if filename=='smoke_ddp.py':text=text.replace('p.obs_encoder.scene_encoder.backbone.model','getattr(p.obs_encoder.scene_encoder.backbone,"model",p.obs_encoder.scene_encoder.backbone)').replace('frozen_b0_no_grad','frozen_backbone_no_grad')
                ast.parse(text);(C/filename).write_text(text)
            relay='#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\nexport https_proxy=http://127.0.0.1:17897 http_proxy=http://127.0.0.1:17897 NO_PROXY=localhost,127.0.0.1\n'
            relay+=f'export TMPDIR={P["eval_tmpdir"]}_wandb\nmkdir -p "$TMPDIR"\nexec {S}/.venv/bin/python -u {C}/wandb_relay.py >> {C}/wandb_relay.log 2>&1\n'
            (C/'start_wandb_relay.sh').write_text(relay)
            roots.append(str(C));cases.append(dict(root=str(C),group=str(G),**spec,machine=machine))
        GP=dict(machine=machine,job_id=job,request_id=request,snapshot=str(S),cases=roots)
        (G/'plan.json').write_text(json.dumps(GP,indent=2))
        text=(OLD/'nodes/machine1/eval_pool.py').read_text()
        text=text.replace("('beat_block_hammer','demo_clean','fm',4,'eager','bf16','RGB_no_flip')","(C['task'],'demo_clean','fm',4,'eager','bf16','RGB_no_flip')")
        text=text.replace("'--task','beat_block_hammer'","'--task',C['task']").replace("task='beat_block_hammer'","task=C['task']")
        ast.parse(text);(G/'eval_pool.py').write_text(text)
        node_groups.append(str(G));groups.append(dict(root=str(G),**GP))
    NP=dict(machine=machine,job_id=job,request_id=request,snapshot=str(S),groups=node_groups)
    (N/'plan.json').write_text(json.dumps(NP,indent=2));(N/'run_node.py').write_bytes((H/'run_node.py').read_bytes())
    task=f'#!/usr/bin/env bash\nset -euo pipefail\nexport S1_ROOT={S} PYTHONPATH={S}/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\nexec {S}/.venv/bin/python -u {N}/run_node.py {N}\n'
    (N/'task.sh').write_text(task);nodes.append(dict(root=str(N),**NP,task_sha256=sha(N/'task.sh')))
plan=dict(campaign=NAME,commit=COMMIT,snapshot=str(S),source_hashes=hashes,overlay=overlay,cases=cases,nodes=nodes,groups=groups,runtime_links=links,authorized=True,norm='groot')
(R/'plan.json').write_text(json.dumps(plan,indent=2));(R/'SEALED').write_text(sha(R/'plan.json'))
print(json.dumps(dict(root=str(R),cases=len(cases),source_files=len(hashes),nodes=[{k:n[k] for k in ['machine','job_id','request_id','task_sha256']} for n in nodes]),indent=2))
