"""Native TensorRT/CUDA runtime: no PyTorch import or execution in this file."""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import time

import numpy as np
import tensorrt as trt
from cuda import cudart as cu

LOGGER = trt.Logger(trt.Logger.WARNING)
QDQ_RECIPE = 'entropy2_w8a8_static_stride2_fp16_exact_avgpool_v4'


def cuda(result):
    err,*values=result
    if int(err):
        raise RuntimeError(f'CUDA error: {err}')
    return values[0] if len(values)==1 else tuple(values)


def smi(fields):
    return subprocess.check_output(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],text=True).strip()


def pids():
    return sorted(subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split())


def exclusive(expected):
    observed=pids()
    assert not set(observed)-set(expected),f'GPU sharing: expected={expected}, observed={observed}'
    if observed!=expected:
        print('PROCESS_CHECK',dict(expected=expected,observed=observed),flush=True)
        time.sleep(.2)
        observed=pids()
    assert not set(observed)-set(expected),f'GPU sharing: expected={expected}, observed={observed}'


class Buffer:
    def __init__(self,shape,dtype=np.float32):
        self.shape=tuple(int(x) for x in shape)
        self.dtype=np.dtype(dtype)
        self.bytes=int(np.prod(self.shape))*self.dtype.itemsize
        self.ptr=int(cuda(cu.cudaMalloc(max(1,self.bytes))))

    def put(self,array,stream=None):
        array=np.ascontiguousarray(array,dtype=self.dtype)
        assert array.shape==self.shape,(array.shape,self.shape)
        cuda(cu.cudaMemcpy(self.ptr,array.ctypes.data,self.bytes,cu.cudaMemcpyKind.cudaMemcpyHostToDevice))

    def get(self):
        arr=np.empty(self.shape,dtype=self.dtype)
        cuda(cu.cudaMemcpy(arr.ctypes.data,self.ptr,self.bytes,cu.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return arr

    def close(self):
        if self.ptr:
            cuda(cu.cudaFree(self.ptr))
            self.ptr=0


class Calibration(trt.IInt8EntropyCalibrator2):
    def __init__(self,root,model,scope,batch,cache):
        super().__init__()
        self.root,self.model,self.scope,self.batch,self.cache=root,model,scope,batch,cache
        self.index=0
        self.count=512 if scope=='step' else 128
        self.arrays={}
        def arr(key,local=False):
            k=(key,local)
            if k not in self.arrays:
                self.arrays[k]=np.load((root/model if local else root)/f'cal_{key}.npy',mmap_mode='r')
            return self.arrays[k]
        self.arr=arr
        sample=self.samples(0)
        self.buffers={k:Buffer(v.shape,v.dtype) for k,v in sample.items()}

    def samples(self,start):
        b=self.batch
        ids=np.arange(start,start+b)%128
        if self.scope=='backbone':
            return {'images':np.asarray(self.arr('processed')[ids])}
        if self.scope=='encoder':
            return {'images':np.asarray(self.arr('images')[ids]),'state':np.asarray(self.arr('state')[ids])}
        # Cover both samplers and every timestep with actual FP32 FM trajectories.
        ids=(np.arange(start,start+b)*17)%128
        phases=np.arange(start,start+b)%14
        nsteps=np.where(phases<4,4,10)
        phases=np.where(phases<4,phases,phases-4)
        noisy=np.stack([self.arr(f'noisy_{n}_{i}',True)[j] for n,i,j in zip(nsteps,phases,ids)])
        return dict(noisy=noisy,normalized=np.asarray(self.arr('normalized',True)[ids]),
                    kv=np.asarray(self.arr('kv',True)[ids]),
                    timestep=(phases/nsteps*1000).astype(np.int64),
                    dt=(1/nsteps).astype(np.float32).reshape(b,1,1))

    def get_batch_size(self):
        return self.batch

    def get_batch(self,names):
        if self.index>=self.count:
            return None
        values=self.samples(self.index)
        for k,v in values.items():
            self.buffers[k].put(v)
        self.index+=self.batch
        if self.index%64==0:
            print('CALIBRATE',self.model,self.scope,self.batch,self.index,'/',self.count,flush=True)
        return [self.buffers[k].ptr for k in names]

    def read_calibration_cache(self):
        return self.cache.read_bytes() if self.cache.exists() else None

    def write_calibration_cache(self,cache):
        self.cache.write_bytes(cache)

    def close(self):
        for b in self.buffers.values():
            b.close()


def engine_path(root,model,scope,precision,batch):
    if scope=='post':
        precision='fp32'
    return root/model/f'{scope}_{precision}_bs{batch}.plan'


def explicit_qdq(onnx_path,cache_path):
    """Use measured entropy scales; W8 per-channel, A8 per-tensor.

    Normalize/softmax and activation-only attention MatMuls retain floating point.
    The same calibration scales are used for BS1 and BS16.
    """
    import onnx
    from onnx import helper,numpy_helper
    target=onnx_path.with_name(onnx_path.stem+'_qdq_v4.onnx')
    if target.exists():
        return target
    scales={}
    for line in cache_path.read_text().splitlines()[1:]:
        key,encoded=line.rsplit(':',1)
        scale=struct.unpack('>f',bytes.fromhex(encoded.strip()))[0]
        if np.isfinite(scale) and scale>0:
            scales[key]=scale
    model=onnx.load(str(onnx_path))
    # The two stride2 adapter blocks map 8x10 -> 4x5 -> 2x3. Adaptive
    # average pooling 2x3 -> 2x2 is EXACTLY AveragePool(kernel=1x2,stride=1).
    # Slice/ReduceMean export triggered pathological TRT10.9 Q/DQ fusion.
    prefix='/model_adapter/patch_merging/'
    pool_nodes=[n for n in model.graph.node if n.name.startswith(prefix) and n.op_type in ['Slice','ReduceMean','Concat']]
    if pool_nodes:
        assert sum(n.op_type=='ReduceMean' for n in pool_nodes)==8
        assert sum(n.op_type=='Slice' for n in pool_nodes)==6
        transpose=next(n for n in model.graph.node if n.name==prefix+'Transpose')
        original_output=transpose.input[0]
        original_input=next(n for n in pool_nodes if n.op_type=='Slice').input[0]
        replaced=[]
        inserted=False
        for n in model.graph.node:
            if n.name.startswith(prefix) and n.op_type in ['Slice','ReduceMean','Concat']:
                if not inserted:
                    replaced.append(helper.make_node('AveragePool',[original_input],[original_output],
                        name=prefix+'ExactAveragePool',kernel_shape=[1,2],strides=[1,1],pads=[0,0,0,0],count_include_pad=0))
                    inserted=True
                continue
            replaced.append(n)
        model.graph.ClearField('node')
        model.graph.node.extend(replaced)
    initializers={x.name:x for x in model.graph.initializer}
    nodes=[]
    added=[]
    activation_map={}
    quantized=[]
    skipped=[]
    for node in model.graph.node:
        if node.op_type not in ['Conv','MatMul','Gemm'] or len(node.input)<2 or node.input[1] not in initializers:
            nodes.append(node)
            continue
        attrs={a.name:helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type=='Conv' and len(node.input)==2 and list(attrs.get('strides',[1,1]))==[2,2]:
            skipped.append(dict(name=node.name,reason='FP16 static-weight fallback: TRT 10.9 treats quantized bias-free stride2 Conv as FP32 dynamic filter'))
            nodes.append(node)
            continue
        x,w=node.input[:2]
        if x not in scales:
            skipped.append(dict(name=node.name,input=x))
            nodes.append(node)
            continue
        if x not in activation_map:
            prefix=f'ptq_activation_{len(activation_map)}'
            scale_name,zero_name=prefix+'_scale',prefix+'_zero'
            added.extend([numpy_helper.from_array(np.array(scales[x],np.float32),scale_name),
                          numpy_helper.from_array(np.array(0,np.int8),zero_name)])
            q,dq=prefix+'_q',prefix+'_dq'
            nodes.extend([helper.make_node('QuantizeLinear',[x,scale_name,zero_name],[q],name=prefix+'_Quantize'),
                          helper.make_node('DequantizeLinear',[q,scale_name,zero_name],[dq],name=prefix+'_Dequantize')])
            activation_map[x]=dq
        weights=numpy_helper.to_array(initializers[w])
        axis=0 if node.op_type=='Conv' else 1
        if node.op_type=='Gemm':
            axis=0 if attrs.get('transB',0) else 1
        axes=tuple(i for i in range(weights.ndim) if i!=axis)
        ws=np.maximum(np.max(np.abs(weights),axis=axes)/127,np.float32(1e-8)).astype(np.float32)
        broad=[1]*weights.ndim
        broad[axis]=weights.shape[axis]
        qw=np.clip(np.rint(weights/ws.reshape(broad)),-127,127).astype(np.int8)
        prefix=f'ptq_weight_{len(quantized)}'
        added.extend([numpy_helper.from_array(qw,prefix+'_int8'),numpy_helper.from_array(ws,prefix+'_scale'),
                      numpy_helper.from_array(np.zeros_like(ws,dtype=np.int8),prefix+'_zero')])
        nodes.append(helper.make_node('DequantizeLinear',[prefix+'_int8',prefix+'_scale',prefix+'_zero'],
                                      [prefix+'_dq'],axis=axis,name=prefix+'_Dequantize'))
        node.input[0]=activation_map[x]
        node.input[1]=prefix+'_dq'
        nodes.append(node)
        # A Q boundary at a bias-free Conv output is needed for TRT to select
        # an INT8 convolution instead of a runtime FP32 dynamic-filter Conv.
        if node.op_type=='Conv' and len(node.input)==2 and node.output[0] in scales:
            original=node.output[0]
            node.output[0]=original+'_pre_output_quant'
            oprefix=prefix+'_output'
            added.extend([numpy_helper.from_array(np.array(scales[original],np.float32),oprefix+'_scale'),
                          numpy_helper.from_array(np.array(0,np.int8),oprefix+'_zero')])
            nodes.extend([helper.make_node('QuantizeLinear',[node.output[0],oprefix+'_scale',oprefix+'_zero'],
                                           [oprefix+'_q'],name=oprefix+'_Quantize'),
                          helper.make_node('DequantizeLinear',[oprefix+'_q',oprefix+'_scale',oprefix+'_zero'],
                                           [original],name=oprefix+'_Dequantize')])
        quantized.append(dict(name=node.name,op=node.op_type,axis=axis,activation_scale=scales[x]))
    model.graph.ClearField('node')
    model.graph.node.extend(nodes)
    used={x for n in nodes for x in n.input}
    model.graph.ClearField('initializer')
    model.graph.initializer.extend([x for x in list(initializers.values())+added if x.name in used])
    onnx.save(model,str(target))
    onnx.checker.check_model(str(target))
    target.with_suffix('.quantization.json').write_text(json.dumps(dict(
        algorithm='PTQ Entropy2 activation scales, symmetric A8 per-tensor / W8 per-channel; Conv and weighted MatMul/Gemm',
        recipe=QDQ_RECIPE,cache=str(cache_path),quantized=quantized,skipped=skipped),indent=2))
    print('EXPLICIT_QDQ',target,len(quantized),'weighted nodes;',len(skipped),'FP fallback nodes',flush=True)
    return target


def build(root,model,scope,precision,batch,calibration_only=False):
    path=engine_path(root,model,scope,precision,batch)
    if path.exists():
        meta=path.with_suffix('.build.json')
        if precision!='int8' or scope=='post' or (meta.exists() and json.loads(meta.read_text()).get('quantization_recipe')==QDQ_RECIPE):
            return path
        # Retain the calibration-only implicit engine as evidence, never label it INT8.
        archived=path.with_name(path.stem+f'_superseded_{time.time_ns()}.plan')
        path.rename(archived)
        for suffix in ['.layers.json','.build.json']:
            if path.with_suffix(suffix).exists():
                path.with_suffix(suffix).rename(archived.with_suffix(suffix))
    onnx_path=root/model/f'{scope}.onnx'
    cache_path=root/model/f'{scope}_int8_bs1.calibration'
    explicit=precision=='int8' and scope!='post' and not calibration_only
    if explicit:
        if not cache_path.exists():
            build(root,model,scope,precision,1,calibration_only=True)
            # Archive the non-quantized calibration build before writing the actual engine.
            if batch==1:
                return build(root,model,scope,precision,batch)
        onnx_path=explicit_qdq(onnx_path,cache_path)
    print('BUILD',path,flush=True)
    builder=trt.Builder(LOGGER)
    net=builder.create_network(0)
    parser=trt.OnnxParser(net,LOGGER)
    assert parser.parse_from_file(str(onnx_path)), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
    # Calibration-only engines are discarded; tactic search is useful only for
    # the final FP16 / explicit-QDQ engines, which retain optimization level 3.
    config.builder_optimization_level=0 if calibration_only else 3
    profile=builder.create_optimization_profile()
    shapes={}
    for i in range(net.num_inputs):
        inp=net.get_input(i)
        shape=tuple(batch if d<0 else d for d in inp.shape)
        shapes[inp.name]=shape
        profile.set_shape(inp.name,shape,shape,shape)
    config.add_optimization_profile(profile)
    calibrator=None
    if scope!='post':
        config.set_flag(trt.BuilderFlag.FP16)
        if precision=='int8' and not explicit:
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator=Calibration(root,model,scope,batch,path.with_suffix('.calibration'))
            config.int8_calibrator=calibrator
            config.set_calibration_profile(profile)
    cache_path=root/model/'tactics.cache'
    config.set_timing_cache(config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b''),False)
    started=time.time()
    plan=builder.build_serialized_network(net,config)
    if calibrator:
        calibrator.close()
    assert plan is not None,f'Build failed {path}'
    path.write_bytes(bytes(plan))
    cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    runtime=trt.Runtime(LOGGER)
    engine=runtime.deserialize_cuda_engine(plan)
    inspector=engine.create_engine_inspector()
    layers=json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    path.with_suffix('.layers.json').write_text(json.dumps(layers,indent=2))
    layer_list=layers.get('Layers',layers if isinstance(layers,list) else [])
    if explicit:
        bad=[l['Name'] for l in layer_list if l.get('HasDynamicFilter')]
        assert not bad,f'Constant weights must not execute as dynamic FP32 filters: {bad}'
    formats=Counter()
    int8_count=0
    for layer in layer_list:
        strings=[str(x.get('Format/Datatype','')) for x in layer.get('Inputs',[])+layer.get('Outputs',[])]
        formats.update(strings)
        if any('int8' in x.lower() for x in strings):
            int8_count+=1
    meta=dict(model=model,scope=scope,precision=precision,batch=batch,
              engine_bytes=path.stat().st_size,build_seconds=time.time()-started,
              inputs=shapes,layers=len(layer_list),layers_touching_int8=int8_count,formats=dict(formats),
              calibration='EntropyCalibrator2, real observations, 512 step / 128 image samples' if calibrator else None,
              explicit_qdq=explicit,
              quantization_recipe=QDQ_RECIPE if explicit else None,
              onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
              plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.build.json').write_text(json.dumps(meta,indent=2))
    print('BUILT',path,meta,flush=True)
    del engine,runtime,inspector,plan,config,parser,net,builder
    gc.collect()
    return path


class Engine:
    def __init__(self,path,batch):
        self.runtime=trt.Runtime(LOGGER)
        self.engine=self.runtime.deserialize_cuda_engine(path.read_bytes())
        assert self.engine
        self.context=self.engine.create_execution_context()
        self.inputs,self.outputs={},{}
        self.owned=[]
        for i in range(self.engine.num_io_tensors):
            key=self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(key)==trt.TensorIOMode.INPUT:
                shape=tuple(batch if d<0 else d for d in self.engine.get_tensor_shape(key))
                assert self.context.set_input_shape(key,shape)
                buf=Buffer(shape,trt.nptype(self.engine.get_tensor_dtype(key)))
                self.inputs[key]=buf
                self.owned.append(buf)
        assert not self.context.infer_shapes()
        for i in range(self.engine.num_io_tensors):
            key=self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(key)==trt.TensorIOMode.OUTPUT:
                buf=Buffer(self.context.get_tensor_shape(key),trt.nptype(self.engine.get_tensor_dtype(key)))
                self.outputs[key]=buf
                self.owned.append(buf)

    def run(self,stream,inputs=None,outputs=None):
        bindings=inputs or self.inputs
        for k in self.inputs:
            assert self.context.set_tensor_address(k,bindings[k].ptr)
        bindings=outputs or self.outputs
        for k in self.outputs:
            assert self.context.set_tensor_address(k,bindings[k].ptr)
        assert self.context.execute_async_v3(stream_handle=stream)

    def close(self):
        for b in self.owned:
            b.close()
        del self.context,self.engine,self.runtime


class Pipeline:
    def __init__(self,root,model,precision,batch):
        self.batch=batch
        self.stream=cuda(cu.cudaStreamCreateWithFlags(cu.cudaStreamNonBlocking))
        self.encoder=Engine(engine_path(root,model,'encoder',precision,batch),batch)
        self.step=Engine(engine_path(root,model,'step',precision,batch),batch)
        self.post=Engine(engine_path(root,model,'post',precision,batch),batch)
        self.backbone=Engine(engine_path(root,model,'backbone',precision,batch),batch)
        self.noise=Buffer((batch,24,14))
        self.buffers=[Buffer((batch,24,14)),Buffer((batch,24,14))]
        self.constants={}
        self.graphs={}
        for steps in [4,10]:
            dt=Buffer((batch,1,1))
            dt.put(np.full((batch,1,1),1/steps,np.float32))
            times=[]
            for i in range(steps):
                t=Buffer((batch,),np.int64)
                t.put(np.full(batch,int(i/steps*1000),np.int64))
                times.append(t)
            self.constants[steps]=(dt,times)

    def put(self,images,state,noise,processed=None):
        self.encoder.inputs['images'].put(images)
        self.encoder.inputs['state'].put(state)
        self.noise.put(noise)
        if processed is not None:
            self.backbone.inputs['images'].put(processed)

    def full(self,steps):
        stream=self.stream
        cuda(cu.cudaMemcpyAsync(self.buffers[0].ptr,self.noise.ptr,self.noise.bytes,cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,stream))
        self.encoder.run(stream)
        dt,times=self.constants[steps]
        for i,t in enumerate(times):
            inputs=dict(noisy=self.buffers[i%2],normalized=self.encoder.outputs['normalized'],
                        kv=self.encoder.outputs['kv'],timestep=t,dt=dt)
            self.step.run(stream,inputs,{'next_noisy':self.buffers[(i+1)%2]})
        self.post.run(stream,dict(noisy=self.buffers[steps%2],state=self.encoder.inputs['state']))

    def capture(self,key,fn):
        for _ in range(3):
            fn()
        cuda(cu.cudaStreamSynchronize(self.stream))
        cuda(cu.cudaStreamBeginCapture(self.stream,cu.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal))
        fn()
        graph=cuda(cu.cudaStreamEndCapture(self.stream))
        executable=cuda(cu.cudaGraphInstantiate(graph,0))
        self.graphs[key]=(graph,executable)
        return lambda: cuda(cu.cudaGraphLaunch(executable,self.stream))

    def close(self):
        cuda(cu.cudaStreamSynchronize(self.stream))
        for graph,executable in self.graphs.values():
            cuda(cu.cudaGraphExecDestroy(executable))
            cuda(cu.cudaGraphDestroy(graph))
        for engine in [self.backbone,self.encoder,self.step,self.post]:
            engine.close()
        for b in [self.noise,*self.buffers]:
            b.close()
        for dt,times in self.constants.values():
            for b in [dt,*times]:
                b.close()
        cuda(cu.cudaStreamDestroy(self.stream))


def metrics(a,b):
    a,b=a.astype(np.float64).ravel(),b.astype(np.float64).ravel()
    assert np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite engine output'
    delta=a-b
    return dict(mae=float(np.abs(delta).mean()),rmse=float(np.sqrt(np.mean(delta**2))),
                max_abs=float(np.abs(delta).max()),
                relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(b),1e-12)),
                cosine=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12)))


def correctness(pipe,root,model):
    b=pipe.batch
    arrays={k:np.load(root/f'eval_{k}.npy',mmap_mode='r') for k in ['images','state','noise','processed','target']}
    actual={key:[] for key in ['tokens','action_4','action_10']}
    for start in range(0,32,b):
        pipe.put(*[np.asarray(arrays[k][start:start+b]) for k in ['images','state','noise','processed']])
        pipe.backbone.run(pipe.stream)
        cuda(cu.cudaStreamSynchronize(pipe.stream))
        actual['tokens'].append(pipe.backbone.outputs['tokens'].get())
        for steps in [4,10]:
            pipe.full(steps)
            cuda(cu.cudaStreamSynchronize(pipe.stream))
            actual[f'action_{steps}'].append(pipe.post.outputs['action'].get())
    result={}
    for key,arrs in actual.items():
        values=np.concatenate(arrs)
        reference=np.load(root/model/f'eval_{key}.npy')
        result[key]=metrics(values,reference)
        if key.startswith('action'):
            result[key]['dataset_error']=metrics(values,np.asarray(arrays['target']))
            result[key]['fp32_dataset_error']=metrics(reference,np.asarray(arrays['target']))
    return result


def measure(fn,stream,repeats=100,warmup=30):
    for _ in range(warmup):
        fn()
    cuda(cu.cudaStreamSynchronize(stream))
    start=cuda(cu.cudaEventCreate())
    end=cuda(cu.cudaEventCreate())
    wall,device=[],[]
    for _ in range(repeats):
        before=time.perf_counter_ns()
        cuda(cu.cudaEventRecord(start,stream))
        fn()
        cuda(cu.cudaEventRecord(end,stream))
        cuda(cu.cudaEventSynchronize(end))
        wall.append((time.perf_counter_ns()-before)/1e6)
        device.append(float(cuda(cu.cudaEventElapsedTime(start,end))))
    cuda(cu.cudaEventDestroy(start))
    cuda(cu.cudaEventDestroy(end))
    return dict(wall=wall,cuda=device)


class LayerProfile(trt.IProfiler):
    def __init__(self):
        super().__init__()
        self.times={}

    def report_layer_time(self,name,ms):
        self.times.setdefault(name,[]).append(ms)


def diagnose_encoder(engine,stream,path):
    """Separate context, after timing: profiling never affects table samples."""
    context=engine.engine.create_execution_context()
    profiler=LayerProfile()
    context.profiler=profiler
    for k,b in engine.inputs.items():
        assert context.set_input_shape(k,b.shape)
        assert context.set_tensor_address(k,b.ptr)
    for k,b in engine.outputs.items():
        assert context.set_tensor_address(k,b.ptr)
    for _ in range(5):
        assert context.execute_async_v3(stream_handle=stream)
        cuda(cu.cudaStreamSynchronize(stream))
    layers=sorted([dict(name=k,mean_ms=float(np.mean(v))) for k,v in profiler.times.items()],key=lambda x:-x['mean_ms'])
    path.write_text(json.dumps(layers,indent=2))
    print('ENCODER_PROFILE_TOP',json.dumps(layers[:5]),flush=True)
    del context,profiler


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--mode',choices=['build','bench'],required=True)
    ap.add_argument('--model',choices=['dino','b2','all'],default='all')
    ap.add_argument('--precision',choices=['fp16','int8','all'],default='all')
    ap.add_argument('--batch',type=int,choices=[1,16])
    ap.add_argument('--rounds',type=int,default=3)
    ap.add_argument('--repeats',type=int,default=100)
    args=ap.parse_args()
    assert 'torch' not in sys.modules, 'Latency runtime must not import PyTorch'
    root=args.root
    models=['dino','b2'] if args.model=='all' else [args.model]
    precisions=['fp16','int8'] if args.precision=='all' else [args.precision]
    batches=[args.batch] if args.batch else [1,16]
    assert not pids(), 'GPU in use by another process'
    cuda(cu.cudaFree(0))
    expected=pids()
    print('GPU_PROCESS',expected,flush=True)
    env=dict(tensorrt=trt.__version__,cuda_runtime=cuda(cu.cudaRuntimeGetVersion()),
             gpu=smi('name,uuid,driver_version,power.limit'),tf32=False,
             clocks_locked=False,rounds=args.rounds,repeats=args.repeats,warmup=30,
             timer='single in-flight synchronous wall / CUDA event',
             batch_definition='one batch item = three RGB views plus state',
             full_scope='GPU raw RGB + state + pre-generated noise -> unnormalized 23x14 actions; preprocess, backbone, fusion, compression, scene adaptor, KV once, FM, postprocessing',
             excludes=['H2D/D2H','video decoding','camera/network','noise RNG','build/calibration/warmup/capture'])
    (root/'runtime_environment.json').write_text(json.dumps(env,indent=2))
    records_path=root/'measurements.jsonl'
    for model in models:
        assert (root/model/'EXPORTED').exists()
        for batch in batches:
            for precision in precisions:
                if args.mode=='build':
                    for scope in ['backbone','encoder','step','post']:
                        exclusive(expected)
                        build(root,model,scope,precision,batch)
                    continue
                stamp=root/model/f'{precision}_bs{batch}_BENCHMARKED'
                if stamp.exists():
                    continue
                if precision=='int8':
                    for scope in ['backbone','encoder','step']:
                        info=json.loads(engine_path(root,model,scope,precision,batch).with_suffix('.build.json').read_text())
                        assert info.get('quantization_recipe')==QDQ_RECIPE and info['layers_touching_int8']>0, f'Actual INT8 execution required: {model}/{scope}/bs{batch}'
                pipe=Pipeline(root,model,precision,batch)
                checks=correctness(pipe,root,model)
                (root/model/f'{precision}_bs{batch}.correctness.json').write_text(json.dumps(checks,indent=2))
                images=np.load(root/'eval_images.npy',mmap_mode='r')[:batch]
                state=np.load(root/'eval_state.npy',mmap_mode='r')[:batch]
                noise=np.load(root/'eval_noise.npy',mmap_mode='r')[:batch]
                processed=np.load(root/'eval_processed.npy',mmap_mode='r')[:batch]
                pipe.put(images,state,noise,processed)
                cases=[]
                for scope,fn in [('backbone',lambda:pipe.backbone.run(pipe.stream)),
                                 ('fm4',lambda:pipe.full(4)),('fm10',lambda:pipe.full(10))]:
                    cases.append((scope,False,fn))
                    cases.append((scope,True,pipe.capture(scope,fn)))
                # Graph output must respond to changed real inputs and match uncaptured TRT.
                graph_checks={}
                for offset in [0,16]:
                    sl=slice(offset,offset+batch)
                    pipe.put(np.load(root/'eval_images.npy',mmap_mode='r')[sl],
                             np.load(root/'eval_state.npy',mmap_mode='r')[sl],
                             np.load(root/'eval_noise.npy',mmap_mode='r')[sl],
                             np.load(root/'eval_processed.npy',mmap_mode='r')[sl])
                    for scope in ['backbone','fm4','fm10']:
                        plain=next(fn for s,g,fn in cases if s==scope and not g)
                        graph=next(fn for s,g,fn in cases if s==scope and g)
                        out=pipe.backbone.outputs['tokens'] if scope=='backbone' else pipe.post.outputs['action']
                        plain();cuda(cu.cudaStreamSynchronize(pipe.stream));a=out.get()
                        graph();cuda(cu.cudaStreamSynchronize(pipe.stream));z=out.get()
                        np.testing.assert_allclose(a,z,rtol=1e-5,atol=1e-5)
                        graph_checks[f'{scope}_{offset}']=metrics(z,a)
                (root/model/f'{precision}_bs{batch}.graph_correctness.json').write_text(json.dumps(graph_checks,indent=2))
                for round_index in range(args.rounds):
                    order=cases if round_index%2==0 else list(reversed(cases))
                    for scope,graph,fn in order:
                        exclusive(expected)
                        row=dict(model=model,batch=batch,precision=precision,scope=scope,graph=graph,round=round_index,
                                 before=smi('temperature.gpu,clocks.current.sm,pstate,power.draw'))
                        row.update(measure(fn,pipe.stream,args.repeats))
                        row['after']=smi('temperature.gpu,clocks.current.sm,pstate,power.draw')
                        with records_path.open('a') as f:
                            f.write(json.dumps(row)+'\n')
                        print('MEASURE',model,batch,precision,scope,graph,round_index,
                              'wall p50/p95',np.percentile(row['wall'],[50,95]),
                              'gpu p50/p95',np.percentile(row['cuda'],[50,95]),flush=True)
                try:
                    diagnose_encoder(pipe.encoder,pipe.stream,root/model/f'{precision}_bs{batch}.layer_profile.json')
                except Exception as exc:
                    print('OPTIONAL_LAYER_PROFILE_FAILED',repr(exc),flush=True)
                pipe.close()
                del pipe
                gc.collect()
                stamp.write_text('OK\n')
    print('COMPLETED',args.mode,flush=True)


if __name__=='__main__':
    main()
