"""Finite build hot queue and one dedicated GPU for comparable policy timings."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name+f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2));tmp.replace(path)


def claim(queue):
    for path in sorted((queue/'pending').glob('*.json')):
        target = queue/'claimed'/path.name
        try:
            path.rename(target)
            return target
        except FileNotFoundError:
            pass
    return None


def worker(args, plan):
    root = Path(plan['root']);source = Path(plan['snapshot'])
    queue = root/('build_queue' if args.role == 'build' else 'bench_queue')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(source/'src'),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    script = source/'tools/quantization/policy_fp16_components.py'
    while not (root/'STOP_WORKERS').exists():
        item = claim(queue)
        if item is None:
            time.sleep(3);continue
        case = read(item);out=Path(plan['artifacts'])/case['id']
        log = root/'logs'/f'{args.role}-{case["id"]}.log'
        try:
            if args.role == 'build':
                modes = ([] if (out/'export_manifest.json').exists() else ['export'])+['build']
            else:
                modes = ['parity', 'bench']
            for mode in modes:
                cmd=[sys.executable,str(script),'--plan',str(args.plan),'--case',case['id'],'--mode',mode]
                with log.open('a') as f:
                    f.write('\nCOMMAND '+repr(cmd)+'\n');f.flush()
                    subprocess.run(cmd,cwd=source,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
                if mode == 'parity':
                    result=read(out/'parity.json')
                    if not result['action_basic_tolerance']:
                        raise RuntimeError('Action parity exceeded inherited RMSE .01 / max .05 numerical check; retain artifacts and diagnose')
                    if case['id']=='stage4_rms_nopool':
                        save(root/'pilot_validated.json',dict(case=case['id'],gpu=args.gpu,time=time.time()))
            if args.role == 'build':
                save(root/'bench_queue/pending'/item.name,case)
            item.rename(queue/'done'/item.name)
            print('DONE',args.role,case['id'],flush=True)
        except Exception:
            with log.open('a') as f:traceback.print_exc(file=f)
            item.rename(queue/'failed'/item.name)
            print('FAILED',args.role,case['id'],flush=True)


def controller(args, plan):
    root=Path(plan['root']);(root/'logs').mkdir(exist_ok=True)
    for q in ['build_queue','bench_queue']:
        for state in ['pending','claimed','done','failed']:(root/q/state).mkdir(parents=True,exist_ok=True)
    pilot=next(c for c in plan['cases'] if c['id']=='stage4_rms_nopool')
    save(root/'build_queue/pending'/f'{pilot["id"]}.json',pilot)
    children=[]
    for role,gpu in [('bench',0)]+[('build',i) for i in range(1,5)]:
        log=(root/'logs'/f'worker-{role}-{gpu}.log').open('w')
        cmd=[sys.executable,__file__,'--plan',str(args.plan),'--role',role,'--gpu',str(gpu)]
        children.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
    started=time.time();expanded=False;success=False
    try:
        while True:
            failed=[str(p) for q in ['build_queue','bench_queue'] for p in (root/q/'failed').glob('*.json')]
            if failed:raise RuntimeError('Failed cases: '+repr(failed))
            if any(p.poll() is not None for p,_ in children):raise RuntimeError('A worker exited unexpectedly')
            if (root/'pilot_validated.json').exists() and not expanded:
                for c in plan['cases']:
                    if c['id']!=pilot['id']:save(root/'build_queue/pending'/f'{c["id"]}.json',c)
                expanded=True
            done=len(list((root/'bench_queue/done').glob('*.json')))
            save(root/'status.json',dict(phase='running',done=done,total=4,expanded=expanded,
                                        time=time.time(),elapsed_seconds=time.time()-started))
            if done==4:
                success=True;break
            if time.time()-started>3*3600:raise TimeoutError('Finite quantization job exceeded three hours')
            time.sleep(5)
    finally:
        (root/'STOP_WORKERS').write_text('finite matrix complete or failed\n')
        for p,log in children:
            p.wait();log.close()
        save(root/'status.json',dict(phase='completed' if success else 'failed',time=time.time()))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--role',choices=['controller','build','bench'],default='controller')
    ap.add_argument('--gpu',type=int,default=0)
    args=ap.parse_args();plan=read(args.plan)
    if args.role=='controller':controller(args,plan)
    else:worker(args,plan)


if __name__=='__main__':main()
