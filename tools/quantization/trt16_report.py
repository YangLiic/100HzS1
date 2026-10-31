"""Verify a completed run and publish a markdown-only summary."""
import hashlib,json,os
from pathlib import Path
from statistics import median
Q=Path('/DRoboticsAILab/users/licheng.yang/quantization/beat_hammer_trt16_20261030_v1');DEST=Path('/data/vepfs/licheng.yang/100HzS1/docs/100HzS1/trt16_fp16_latency_parameters_20261030.md')
plan=json.loads((Q/'cases_input.json').read_text()); assert len(plan)==16
assert len(list((Q/'queue_v3/done').glob('*.json')))==16
assert not list((Q/'queue_v3/failed').glob('*.json')) and not list((Q/'queue_v3/pending').glob('*.json')) and not list((Q/'queue_v3/claimed').glob('*.json'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(16<<20):h.update(x)
 return h.hexdigest()
rows=[];versions=set()
for r in plan:
 id=Path(r['result']).parent.name.rsplit('_nfe',1)[0];case=Q/'cases_v2'/id;manifest=json.loads((case/'build_manifest.json').read_text());params=json.loads((case/'parameters.json').read_text());assert manifest['checkpoint']==r['checkpoint'] and sha(Path(r['checkpoint'])/'model.safetensors')==manifest['weight_sha256']
 assert len(manifest['plans'])==4
 for path,digest in manifest['plans'].items():assert sha(case/path)==digest,(id,path)
 assert params['all_nonvisual']==params['action_stack']+params['scene_adapter'] and params['action_stack']==params['model_dit_including_time_fusion']+params['state_encoder']
 rec={}
 for n in ([1] if r['flow']=='MeanFlow' else [1,4]):
  d=json.loads((case/f'latency_nfe{n}.json').read_text());p=json.loads((case/f'parity_nfe{n}.json').read_text());assert d['checkpoint']==r['checkpoint'] and d['backend']=='TensorRT' and d['precision']=='fp16' and d['nfe']==n and d['flow_type']==r['flow'].lower() and d['batch']==1 and d['views']==3 and d['kv_cache'] and not d['graph'] and d['output_finite'];assert d['post_precision']=='fp32' and d['warmup']==30 and d['rounds']==3 and d['repeats']==100
  assert len(d['vision']['gpu_ms']['samples'])==len(d['full']['gpu_ms']['samples'])==300 and p['steps']==n and p['all_finite'] and p['rmse']<=.01 and p['max_abs']<=.05
  versions.add(d['trt_version']);rec[n]=d
 rows.append((r,id,params,rec))
assert len(rows)==16 and sum(len(x[3]) for x in rows)==24
order={'B0':0,'B1':1,'B2':2,'DINOv3 Base':3};rows.sort(key=lambda z:(order[z[0]['backbone']],z[0]['flow']!='FM',z[0]['injection']!='tau_token'))
f=lambda x:f'{x:.3f}'
L=['# beat_block_hammer：16 个策略的 TensorRT FP16 延迟与参数量','', '## 测量口径','',f'- NVIDIA L4，batch=1，单个样本包含 3 路 RGB（原始 240×320）及 1 个 state；TensorRT {", ".join(sorted(versions))}，KV cache 开启，CUDA Graph 关闭。','- FP16 表示 backbone、encoder 和 step 引擎启用 TensorRT FP16 builder flag，允许混合精度；后处理引擎保留 FP32。','- 视觉编码：三视图经训练一致的预处理后，仅计 backbone TensorRT 引擎；不含图像预处理、视角融合和压缩。完整推理：GPU 上原始 RGB、state、预生成噪声进入，包含预处理、视觉 backbone、融合、压缩、条件投影、KV 构造、DiT 各步和动作后处理。','- 延迟为 CUDA event 计时的单次同步 p50（ms）；每模型每步数预热 30 次、采样 3×100 次。输入已在 GPU，不含 H2D/D2H、采图、视频解码、网络传输、噪声生成、构建、加载及预热；不是端到端机器人控制延迟。','- 参数按独立 PyTorch 参数对象计数，单位 M；动作部分 = DiT（含时间编码、AdaRMS 和 action-time fusion）+ state_adaptor。非视觉总计另包含 scene_adaptor；不包含 obs_encoder（视觉 backbone、视角融合和视觉压缩），也不计 normalizer 统计。', '- 每行是各自训练后的权重，不是简单地对同一个权重替换时间注入；参数量是架构计数，FP16 并不会改变参数个数。','', '## 结果','', '| Backbone | 方法 | 注入 | 视觉 backbone (ms) | 完整 1-step (ms) | 完整 4-step (ms) | DiT + state (M) | scene adaptor (M) | 非视觉总计 (M) |','|---|---|---|---:|---:|---:|---:|---:|---:|']
for r,id,p,d in rows:
 x=d[1];v=f(x['vision']['gpu_ms']['p50']);one=f(x['full']['gpu_ms']['p50']);four=f(d[4]['full']['gpu_ms']['p50']) if 4 in d else '—';L.append(f"| {r['backbone']} | {r['flow']} | {r['injection']} | {v} | {one} | {four} | {p['action_stack']/1e6:.3f} | {p['scene_adapter']/1e6:.3f} | {p['all_nonvisual']/1e6:.3f} |")
L+=['','## 数值校验与产物','', '- 每个模型与步数均使用相同生成的 RGB、state 和噪声，对比 TensorRT 输出与 PyTorch FP32 eager 输出；要求全部有限、RMSE ≤ 0.01、最大绝对误差 ≤ 0.05。该检查用于发现导出偏差，不等于真实机器人任务精度验证。','- 引擎构建时 TensorRT 对部分 FP16 归一化给出潜在溢出提示；本报告保留 FP32 后处理，且以上校验全部通过，但不能保证所有输入都无数值风险。',f'- 原始计划、逐次时延、校验数据、参数明细和日志：`{Q}`。模型权重及各次闭环评测来源见该目录 `cases_input.json`。','- 先前 100 episode 闭环结果基于 PyTorch BF16；不能视为本次 TensorRT FP16 的闭环成功率。']
assert not DEST.exists(),f'report exists; refusing overwrite: {DEST}'
DEST.parent.mkdir(parents=True,exist_ok=True);tmp=DEST.with_name(DEST.name+'.tmp');tmp.write_text('\n'.join(L)+'\n');os.replace(tmp,DEST)
print('REPORT_READY',DEST)
