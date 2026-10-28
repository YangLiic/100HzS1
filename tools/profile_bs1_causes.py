"""Explain L4 image-bs1 latency using operator traces and controlled diagnostics."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from transformers import AutoConfig
from vfa.models.policy.vfa.emvis.encoder.dinov3cnn_encoder import DINOv3Encoder
from vfa.models.policy.vfa.emvis.encoder.efficientvit_encoder import EfficientViTEncoder

OUT=Path('/DRoboticsAILab/users/licheng.yang/robotwin2.0/VFA/benchmarks/20261028_bs1_cause_profile_v1')


def pids():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return sorted(s.strip() for s in text.splitlines() if s.strip())


def measure(function,repeats=200):
    for _ in range(30):function()
    torch.cuda.synchronize()
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record();end.record();end.synchronize()
    wall,device=[],[]
    for _ in range(repeats):
        start=time.perf_counter();begin.record();function();end.record();end.synchronize()
        wall.append((time.perf_counter()-start)*1000);device.append(begin.elapsed_time(end))
    return dict(wall_p50_ms=float(np.median(wall)),wall_p95_ms=float(np.percentile(wall,95)),
                cuda_p50_ms=float(np.median(device)),samples=repeats)


def profile(function,name):
    for _ in range(20):function()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(5):function()
        torch.cuda.synchronize()
    gpu=[e for e in prof.events() if str(e.device_type).endswith('CUDA')]
    kernels=Counter(e.name for e in gpu)
    operators=[]
    for e in prof.key_averages():
        operators.append(dict(name=e.key,calls_per_forward=e.count/5,
                              self_cpu_us_per_forward=e.self_cpu_time_total/5,
                              self_device_us_per_forward=e.self_device_time_total/5))
    prof.export_chrome_trace(str(OUT/f'{name}.trace.json'))
    return dict(gpu_events_per_forward=len(gpu)/5,
                gpu_event_time_sum_ms_per_forward=sum(e.device_time_total for e in gpu)/5000,
                top_gpu_kernels=[dict(name=k,calls_per_forward=v/5) for k,v in kernels.most_common(20)],
                operators=sorted(operators,key=lambda r:r['self_device_us_per_forward'],reverse=True))


def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    assert not pids(),'GPU occupied; no diagnostics launched'
    torch.set_num_threads(1);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    path='/data/vepfs/licheng.yang/VFA/pretrained/dinov3-cnn-base'
    dino=DINOv3Encoder(AutoConfig.from_pretrained(path,local_files_only=True))
    dino.load_pretrained_model(path,freeze=True);dino.cuda().eval()
    b2=EfficientViTEncoder('efficientvit_b2.r256_in1k',frozen=True)
    b2.load_pretrained_model('/DRoboticsAILab/users/licheng.yang/pretrained/efficientvit/efficientvit_b2.r256_in1k')
    b2.cuda().eval()
    expected=pids();assert len(expected)==1
    x=torch.rand(1,3,256,320,device='cuda')
    x=(x-x.new_tensor([.485,.456,.406])[None,:,None,None])/x.new_tensor([.229,.224,.225])[None,:,None,None]
    result={}
    def spatial(model,value):
        if isinstance(model,EfficientViTEncoder):return model.model.forward_features(value)
        for stage in model.stages:value=stage(value)
        return value
    for name,model in [('dinov3_base',dino),('efficientvit_b2',b2)]:
        assert pids()==expected
        inventory=Counter(type(m).__name__ for m in model.modules())
        value=dict(module_inventory=dict(inventory))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            fn=lambda m=model:spatial(m,x)
            value['eager_fp32_weights_bf16_autocast']=measure(fn)
            value['profile_fp32_weights_bf16_autocast']=profile(fn,name)
            reference=fn().clone()
            # Static-shape launch diagnostic, not a change to the production model.
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(20):fn()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                with torch.cuda.graph(graph): captured=fn()
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(captured,reference,rtol=1e-5,atol=1e-5)
            value['cuda_graph_replay']=measure(graph.replay)
            value['cuda_graph_max_abs_output_difference']=float((captured-reference).abs().max())
        converted=deepcopy(model).bfloat16().eval()
        low_input=x.bfloat16()
        with torch.inference_mode():
            fn_low=lambda:spatial(converted,low_input)
            value['eager_native_bf16_weights']=measure(fn_low)
            value['profile_native_bf16_weights']=profile(fn_low,name+'_native_bf16')
            low_out=fn_low().float()
            value['native_bf16_max_abs_output_difference']=float((low_out-reference.float()).abs().max())
        del converted,graph,captured
        result[name]=value
        (OUT/'summary.json').write_text(json.dumps(result,indent=2))
        print('PROFILE_COMPLETE '+json.dumps({k:v for k,v in value.items() if not k.startswith('profile') and k!='module_inventory'}),flush=True)
    assert pids()==expected
    (OUT/'COMPLETED').write_text('Profiling and launch/weight-dtype diagnostics completed.\n')
    print('CAUSE_ANALYSIS_COMPLETED',flush=True)


if __name__=='__main__':main()
