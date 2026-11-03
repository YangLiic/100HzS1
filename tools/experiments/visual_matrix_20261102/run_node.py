"""Eight-GPU training followed by independent four-GPU evaluation lanes per task."""
from pathlib import Path
import json,sys,subprocess,os,time,traceback
N=Path(sys.argv[1]);P=json.loads((N/'plan.json').read_text());S=Path(P['snapshot']);PY=str(S/'.venv/bin/python')
env=dict(os.environ,S1_ROOT=str(S),PYTHONPATH=str(S/'src'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
failed=[]
def status(phase,**extra):
    row=dict(phase=phase,time=time.time(),**extra);temp=N/'status.tmp';temp.write_text(json.dumps(row,indent=2));temp.replace(N/'status.json');print(json.dumps(row),flush=True)
def run(cmd,log,extra=None):
    e=dict(env,**(extra or {}))
    with log.open('a') as f:
        f.write('COMMAND '+repr(cmd)+'\n');f.flush();return subprocess.run(cmd,cwd=S,env=e,stdout=f,stderr=subprocess.STDOUT).returncode
status('starting')
for group in P['groups']:
    G=Path(group);GP=json.loads((G/'plan.json').read_text());ready=[]
    for root in GP['cases']:
        C=Path(root);case=json.loads((C/'plan.json').read_text());status('preflight',case=case['case'])
        if not (C/'preflight.json').exists():
            code=run([PY,str(C/'preflight.py'),str(C)],C/'preflight.log')
            if code:failed.append(dict(case=case['case'],stage='preflight',code=code));(C/'FAILED.json').write_text(json.dumps(failed[-1]));continue
        status('startup_check',case=case['case'])
        code=run([PY,str(C/'launch.py'),'--smoke-only'],C/'launch_smoke.log',{'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
        if not code:
            status('training',case=case['case']);code=run([PY,str(C/'launch.py')],C/'launch_train.log',{'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
        if code:failed.append(dict(case=case['case'],stage='training_or_strict',code=code));(C/'FAILED.json').write_text(json.dumps(failed[-1]));continue
        ready.append(root)
    # Failed cases are preserved. Their pending eval items remain outside the ready queue.
    if ready:
        missing=set(GP['cases'])-set(ready)
        if missing:
            (G/'queue/blocked').mkdir(exist_ok=True)
            for f in (G/'queue/pending').glob('*.json'):
                if json.loads(f.read_text())['case_plan'].removesuffix('/plan.json') in missing:f.rename(G/'queue/blocked'/f.name)
            GP['cases']=ready;(G/'ready_plan.json').write_text(json.dumps(GP,indent=2))
            # The original group manifest stays immutable; run a separate ready manifest directory.
            runner=(G/'eval_pool.py').read_text().replace("P=json.loads((N/'plan.json').read_text())","P=json.loads((N/'ready_plan.json').read_text())")
            (G/'eval_ready.py').write_text(runner);script=G/'eval_ready.py'
        else:script=G/'eval_pool.py'
        status('evaluating',group=str(G));code=run([PY,str(script),str(G)],G/'evaluation.log')
        if code:failed.append(dict(group=str(G),stage='evaluation',code=code))
(N/'failures.json').write_text(json.dumps(failed,indent=2));status('failed' if failed else 'completed',failures=failed)
sys.exit(1 if failed else 0)
