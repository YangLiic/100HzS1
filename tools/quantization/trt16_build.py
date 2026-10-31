"""Build 100HzS1 FP16 TensorRT artifacts for one checkpoint; never edits checkpoint."""
import argparse, gc, hashlib, importlib, json, math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

class Backbone(nn.Module):
    def __init__(self, model):
        super().__init__(); self.backbone=model
    def forward(self, x):
        return self.backbone(x)['image_tokens']

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(16<<20): h.update(data)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    exp=importlib.import_module('100hzs1.inference.export_tensorrt')
    load=importlib.import_module('100hzs1.models.base_runner').load_policy
    p=load(str(args.checkpoint),device='cpu').float().eval()
    # Count unique parameter objects; buffers (normalizer statistics and masks) excluded.
    seen=set(); groups={}
    for name,parameter in p.named_parameters():
        if id(parameter) in seen:continue
        seen.add(id(parameter));group=name.split('.')[0];groups[group]=groups.get(group,0)+parameter.numel()
    details={}
    for name,parameter in p.model.named_parameters():
        group=name.split('.')[0];details[group]=details.get(group,0)+parameter.numel()
    expected={'model','state_adaptor','scene_adaptor','obs_encoder','normalizer'}
    assert set(groups).issubset(expected),(groups.keys(),expected)
    counts={'model_dit_including_time_fusion':groups.get('model',0),'state_encoder':groups.get('state_adaptor',0),'scene_adapter':groups.get('scene_adaptor',0),'excluded_visual_encoder_fusion_compression':groups.get('obs_encoder',0),'action_stack':groups.get('model',0)+groups.get('state_adaptor',0),'normalizer_statistics_excluded':groups.get('normalizer',0),'all_nonvisual':sum(groups.get(k,0) for k in ('model','state_adaptor','scene_adaptor')),'model_children':details,'unit':'parameters','checkpoint':str(args.checkpoint)}
    (args.output/'parameters.json').write_text(json.dumps(counts,indent=2))
    metadata=exp.metadata_for_export(p,True);height,width=metadata['processed_hw']
    p.to('cuda:0');m=Backbone(p.obs_encoder.scene_encoder.backbone).eval();exp.patch_export()
    x=torch.rand((1,len(p.obs_encoder.rgb_key),3,height,width),device='cuda:0');onnx=args.output/'backbone.onnx'
    with torch.no_grad():torch.onnx.export(m,(x,),str(onnx),opset_version=19,dynamo=False,input_names=['images'],output_names=['tokens'],do_constant_folding=True)
    trt,_=exp.dependencies();logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger);network=builder.create_network(0);parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(onnx)):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg=builder.create_builder_config();cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4<<30);cfg.clear_flag(trt.BuilderFlag.TF32);cfg.builder_optimization_level=3;cfg.set_flag(trt.BuilderFlag.FP16)
    plan=builder.build_serialized_network(network,cfg)
    if plan is None:raise RuntimeError('TensorRT FP16 backbone build failed')
    (args.output/'backbone.plan').write_bytes(bytes(plan));onnx.unlink()
    del plan,parser,network,cfg,builder,m,p,x;gc.collect();torch.cuda.empty_cache()
    bundle=args.output/'bundle';bundle.mkdir(exist_ok=True)
    exp.export_and_build(SimpleNamespace(checkpoint=args.checkpoint,output=bundle,device='cuda:0',precision='fp16',batch_size=1,kv_cache=True))
    manifest={'checkpoint':str(args.checkpoint),'weight_sha256':digest(args.checkpoint/'model.safetensors'),'config_sha256':digest(args.checkpoint/'config.json'),'plans':{str(p.relative_to(args.output)):digest(p) for p in args.output.rglob('*.plan')},'metadata':json.loads((bundle/'metadata.json').read_text())}
    assert len(manifest['plans'])==4
    (args.output/'build_manifest.json').write_text(json.dumps(manifest,indent=2));print('BUILD_COMPLETE',args.output,flush=True)
if __name__=='__main__':main()
