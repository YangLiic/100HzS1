"""True image-batch=1 L4 timing, bare spatial trunks and VFA encoder APIs."""
import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import timm
import transformers
from transformers import AutoConfig

from vfa.models.policy.vfa.emvis.encoder.dinov3cnn_encoder import DINOv3Encoder
from vfa.models.policy.vfa.emvis.encoder.efficientvit_encoder import EfficientViTEncoder


def pids():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return sorted(s.strip() for s in text.splitlines() if s.strip())


def gpu_state():
    return subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,power.limit,temperature.gpu,clocks.current.sm,power.draw','--format=csv,noheader'],text=True).strip()


def stats(values):
    return dict(p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),mean_ms=float(np.mean(values)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=150)
    parser.add_argument('--warmup',type=int,default=30)
    args=parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    assert not pids(),'GPU already occupied; no timing performed'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(42)
    dino_path='/data/vepfs/licheng.yang/VFA/pretrained/dinov3-cnn-base'
    b2_path='/DRoboticsAILab/users/licheng.yang/pretrained/efficientvit/efficientvit_b2.r256_in1k'
    dino=DINOv3Encoder(AutoConfig.from_pretrained(dino_path,local_files_only=True))
    dino.load_pretrained_model(dino_path,freeze=True)
    dino.cuda().eval()
    b2=EfficientViTEncoder('efficientvit_b2.r256_in1k',frozen=True)
    b2.load_pretrained_model(b2_path,freeze=True)
    b2.cuda().eval()
    expected=pids()
    assert len(expected)==1,expected
    inputs={}
    for height,width in ((256,320),(256,256)):
        value=torch.rand(1,3,height,width,device='cuda')
        mean=value.new_tensor([.485,.456,.406]).reshape(1,3,1,1)
        std=value.new_tensor([.229,.224,.225]).reshape(1,3,1,1)
        inputs[height,width]=(value-mean)/std

    def dino_spatial(x):
        for stage in dino.stages:x=stage(x)
        return x

    methods={
        ('dinov3_base','spatial_trunk'):dino_spatial,
        ('efficientvit_b2','spatial_trunk'):b2.model.forward_features,
        ('dinov3_base','vfa_encoder_api'):lambda x:dino(x.unsqueeze(1))['image_tokens'],
        ('efficientvit_b2','vfa_encoder_api'):lambda x:b2(x.unsqueeze(1))['image_tokens'],
    }
    metadata=dict(device=gpu_state(),torch=torch.__version__,cuda=torch.version.cuda,
                  timm=timm.__version__,transformers=transformers.__version__,image_batch=1,
                  weights_dtype='float32',tf32=False,compile=False,cpu_threads=1,
                  warmup_per_case=args.warmup,repeats_per_round=args.repeats,rounds=args.rounds,
                  weights=dict(dinov3_base=dino_path,efficientvit_b2=b2_path),
                  parameters=dict(dinov3_base=sum(p.numel() for p in dino.parameters()),
                                  efficientvit_b2=sum(p.numel() for p in b2.parameters())),
                  definitions=dict(spatial_trunk='DINO stages only / EfficientViT forward_features: raw stride32 feature maps',
                                   vfa_encoder_api='actual one-view VFA encoder, including DINO final normalization and token/position packaging'),
                  exclusions=['resize','normalization','H2D','classification head','VFA projection','view-aware fusion','visual compression','DiT'],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'environment.json').write_text(json.dumps(metadata,indent=2))
    records=[];checks=[]
    for precision,dtype in (('bf16',torch.bfloat16),('fp16',torch.float16)):
        with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
            cases=[]
            for (height,width),x in inputs.items():
                assert tuple(x.shape)==(1,3,height,width)
                for scope in ('spatial_trunk','vfa_encoder_api'):
                    for name in ('dinov3_base','efficientvit_b2'):
                        function=methods[name,scope]
                        output=function(x)
                        assert output.shape[0]==1 and torch.isfinite(output).all()
                        check=dict(model=name,precision=precision,scope=scope,input_shape=list(x.shape),output_shape=list(output.shape))
                        if scope=='spatial_trunk':
                            assert tuple(output.shape)==(1,1024 if name=='dinov3_base' else 384,height//32,width//32)
                        checks.append(check)
                        print('CHECK '+json.dumps(check),flush=True)
                        cases.append((name,scope,height,width,function,x))
            for round_index in range(args.rounds):
                order=list(cases)
                if round_index%2:order.reverse()
                elif round_index:order=order[len(order)//2:]+order[:len(order)//2]
                for name,scope,height,width,function,x in order:
                    assert pids()==expected,'Other GPU compute detected; benchmark invalidated'
                    for _ in range(args.warmup):function(x)
                    torch.cuda.synchronize()
                    # Reuse events: event creation is outside the measured region.
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();end.record();end.synchronize()
                    wall=[];device=[]
                    for _ in range(args.repeats):
                        started=time.perf_counter();begin.record()
                        function(x)
                        end.record();end.synchronize()
                        wall.append((time.perf_counter()-started)*1000)
                        device.append(begin.elapsed_time(end))
                    record=dict(model=name,scope=scope,precision=precision,input_hw=[height,width],image_batch=1,
                                round=round_index,wall_ms=wall,cuda_ms=device,wall_summary=stats(wall),cuda_summary=stats(device),gpu=gpu_state())
                    records.append(record)
                    with (args.output/'samples.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print('MEASURE '+json.dumps({k:v for k,v in record.items() if k not in ('wall_ms','cuda_ms')}),flush=True)
    groups={}
    for r in records:groups.setdefault((r['model'],r['scope'],r['precision'],*r['input_hw']),[]).append(r)
    rows=[]
    for (name,scope,precision,height,width),values in groups.items():
        wall=[v for r in values for v in r['wall_ms']]
        device=[v for r in values for v in r['cuda_ms']]
        rows.append(dict(model=name,scope=scope,precision=precision,input_hw=[height,width],image_batch=1,
                         samples=len(wall),wall=stats(wall),cuda=stats(device),round_p50_ms=[r['wall_summary']['p50_ms'] for r in values]))
    assert pids()==expected
    (args.output/'summary.json').write_text(json.dumps(dict(environment=metadata,checks=checks,measurements=rows),indent=2))
    (args.output/'COMPLETED').write_text('True single-image backbone benchmark finished.\n')
    print('SINGLE_IMAGE_BENCHMARK_COMPLETED',flush=True)


if __name__=='__main__':main()
