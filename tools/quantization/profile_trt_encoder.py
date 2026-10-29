"""Separate per-layer diagnostics; never used as latency table samples."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorrt as trt
from benchmark_trt_latency import Engine, cuda, cu, pids, exclusive, engine_path


class Profile(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.times={}

    def report_layer_time(self,layer_name,ms):
        self.times.setdefault(layer_name,[]).append(ms)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--model',choices=['all','dino','b2'],default='all')
    args=ap.parse_args()
    root=args.root
    assert not pids(),'GPU must be idle before diagnostics'
    cuda(cu.cudaFree(0))
    expected=pids()
    stream=cuda(cu.cudaStreamCreateWithFlags(cu.cudaStreamNonBlocking))
    result=[]
    for model in ['dino','b2'] if args.model=='all' else [args.model]:
        for precision in ['fp16','int8']:
            exclusive(expected)
            engine=Engine(engine_path(root,model,'encoder',precision,1),1)
            for key in engine.inputs:
                engine.inputs[key].put(np.load(root/f'eval_{key}.npy',mmap_mode='r')[16:17])
            for _ in range(5):
                engine.run(stream)
            cuda(cu.cudaStreamSynchronize(stream))
            profiler=Profile()
            engine.context.profiler=profiler
            for _ in range(10):
                engine.run(stream)
                cuda(cu.cudaStreamSynchronize(stream))
            layers=sorted([dict(name=k,mean_ms=float(np.mean(v))) for k,v in profiler.times.items()],key=lambda x:-x['mean_ms'])
            row=dict(model=model,precision=precision,batch=1,
                     profiled_gpu_layer_sum_ms=sum(x['mean_ms'] for x in layers),layers=layers)
            result.append(row)
            (root/'encoder_layer_diagnostics.json').write_text(json.dumps(result,indent=2))
            print(model,precision,row['profiled_gpu_layer_sum_ms'],json.dumps(layers[:8]),flush=True)
            engine.close()
    cuda(cu.cudaStreamDestroy(stream))
    (root/'encoder_layer_diagnostics.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
