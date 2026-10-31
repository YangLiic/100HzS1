"""Claim complete model jobs atomically; build and benchmark serially per L4."""
import json,os,subprocess,sys,traceback
from pathlib import Path
Q=Path('/DRoboticsAILab/users/licheng.yang/quantization/beat_hammer_trt16_20261030_v1');S=Q/'source_snapshot_v3';QUEUE=Q/'queue_v3';worker=sys.argv[1]
for state in ('pending','claimed','done','failed'): (QUEUE/state).mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(PYTHONPATH=str(S/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',TMPDIR=str(Q/'tmp'/worker));Path(env['TMPDIR']).mkdir(parents=True,exist_ok=True)
py=str(S/'.venv/bin/python')
def claim():
 for p in sorted((QUEUE/'pending').glob('*.json')):
  try:p.rename(QUEUE/'claimed'/p.name);return QUEUE/'claimed'/p.name
  except FileNotFoundError:continue
 return None
def run(args,log):
 with log.open('a') as f:
  f.write('\nCOMMAND '+repr(args)+'\n');f.flush();subprocess.run(args,cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
while True:
 p=claim()
 if p is None:break
 try:
  c=json.loads(p.read_text());case=Q/'cases_v2'/c['id'];case.mkdir(parents=True,exist_ok=True);log=Q/'logs'/(c['id']+'.log');print('CASE_START',worker,c['id'],flush=True)
  if not (case/'build_manifest.json').exists():run([py,str(S/'tools/quantization/trt16_build.py'),'--checkpoint',c['checkpoint'],'--output',str(case)],log)
  manifest=json.loads((case/'build_manifest.json').read_text());assert manifest['checkpoint']==c['checkpoint'] and len(manifest['plans'])==4
  for steps in ([1] if c['flow']=='MeanFlow' else [1,4]):
   if not (case/f'latency_nfe{steps}.json').exists():run([py,str(S/'tools/quantization/trt16_benchmark.py'),'--checkpoint',c['checkpoint'],'--case',str(case),'--steps',str(steps)],log)
   if not (case/f'parity_nfe{steps}.json').exists():run([py,str(S/'tools/quantization/trt16_parity.py'),'--checkpoint',c['checkpoint'],'--case',str(case),'--steps',str(steps)],log)
  p.rename(QUEUE/'done'/p.name);print('CASE_COMPLETE',worker,c['id'],flush=True)
 except Exception:
  traceback.print_exc();p.rename(QUEUE/'failed'/p.name);print('CASE_FAILED',worker,p.name,flush=True)
print('WORKER_FINISHED',worker,flush=True)
