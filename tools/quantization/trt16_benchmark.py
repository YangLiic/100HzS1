"""Time native TensorRT plans with CUDA events, one synchronized inference at a time."""
import argparse, importlib, json, time
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--case',type=Path,required=True);ap.add_argument('--steps',type=int,required=True);ap.add_argument('--warmup',type=int,default=30);ap.add_argument('--repeats',type=int,default=100);ap.add_argument('--rounds',type=int,default=3);args=ap.parse_args()
    assert args.steps in (1,4) and args.warmup>0 and args.repeats>0 and args.rounds>0
    tr=importlib.import_module('100hzs1.inference.tensorrt');common=importlib.import_module('100hzs1.common.inference');trt,cu=tr.dependencies();check=tr.checked
    metadata=json.loads((args.case/'bundle/metadata.json').read_text());assert metadata['precision']=='fp16' and metadata['use_kv_cache']
    opts=common.checkpoint_inference_options(args.checkpoint,overrides={'backend':'tensorrt','precision':'fp16','num_inference_timesteps':args.steps,'use_cuda_graph':False,'use_kv_cache':True,'tensorrt_batch_size':1})
    tr.ensure_engines=lambda checkpoint,options,hardware,device:(args.case/'bundle',metadata)
    runtime=tr.TensorRTPolicy(args.checkpoint,'cuda:0',opts);backbone=tr.Engine(args.case/'backbone.plan',1,trt,cu)
    try:
        gen=np.random.default_rng(4231); images=gen.random(runtime.encoder.inputs['images'].shape,dtype=np.float32); state=np.zeros(runtime.encoder.inputs['state'].shape,dtype=np.float32);noise=gen.standard_normal(runtime.noise.shape).astype(np.float32)
        runtime.encoder.inputs['images'].put(images);runtime.encoder.inputs['state'].put(state);runtime.base_state.put(state[:,:1]);runtime.noise.put(noise)
        for b in backbone.inputs.values():b.put(gen.random(b.shape,dtype=np.float32))
        def measure(fn):
            gpu=[];wall=[];start=check(cu.cudaEventCreate());end=check(cu.cudaEventCreate())
            try:
                for _ in range(args.warmup):fn()
                check(cu.cudaStreamSynchronize(runtime.stream))
                for _ in range(args.rounds):
                    for _ in range(args.repeats):
                        t0=time.perf_counter_ns();check(cu.cudaEventRecord(start,runtime.stream));fn();check(cu.cudaEventRecord(end,runtime.stream));check(cu.cudaEventSynchronize(end));wall.append((time.perf_counter_ns()-t0)/1e6);gpu.append(float(check(cu.cudaEventElapsedTime(start,end))))
            finally:check(cu.cudaEventDestroy(start));check(cu.cudaEventDestroy(end))
            a=np.array(gpu);b=np.array(wall);return {'gpu_ms':{'mean':float(a.mean()),'p50':float(np.median(a)),'p95':float(np.percentile(a,95)),'samples':gpu},'wall_ms':{'mean':float(b.mean()),'p50':float(np.median(b)),'p95':float(np.percentile(b,95)),'samples':wall}}
        vision=measure(lambda:backbone.run(runtime.stream));full=measure(lambda:runtime._full())
        action=runtime.post.outputs['action'].get();assert np.isfinite(action).all()
        result={'checkpoint':str(args.checkpoint),'backend':'TensorRT','precision':'fp16','post_precision':json.loads((args.case/'bundle/post.export.json').read_text())['precision'],'nfe':args.steps,'flow_type':metadata['flow_type'],'graph':False,'kv_cache':True,'batch':1,'views':len(metadata['rgb_keys']),'raw_image_shapes':metadata['raw_image_shapes'],'backbone_input_shapes':{k:list(v.shape) for k,v in backbone.inputs.items()},'processed_hw':metadata['processed_hw'],'warmup':args.warmup,'rounds':args.rounds,'repeats':args.repeats,'vision':vision,'full':full,'timing':'CUDA event and synchronized host wall; input already on GPU, no H2D/D2H, RNG, build, capture, decoding or networking','output_finite':True,'action_min':float(action.min()),'action_max':float(action.max()),'trt_version':trt.__version__}
        assert runtime._captures==runtime._replays==0
        (args.case/f'latency_nfe{args.steps}.json').write_text(json.dumps(result,indent=2));print('BENCHMARK_COMPLETE',args.steps,'vision',vision['gpu_ms']['p50'],'full',full['gpu_ms']['p50'],flush=True)
    finally:backbone.close();runtime.close()
if __name__=='__main__':main()
