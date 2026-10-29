"""Validate completeness and write short comparison tables, with raw evidence separate."""
import argparse
from collections import defaultdict
import json
import hashlib
from pathlib import Path

import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    root=args.root
    groups=defaultdict(list)
    records=[json.loads(line) for line in (root/'measurements.jsonl').read_text().splitlines()]
    for row in records:
        groups[(row['model'],row['precision'],row['batch'],row['scope'],row['graph'])].append(row)
    measurements=[]
    index={}
    for model in ['dino','b2']:
        for precision in ['fp16','int8']:
            for batch in [1,16]:
                assert (root/model/f'{precision}_bs{batch}_BENCHMARKED').exists()
                for scope in ['backbone','fm4','fm10']:
                    for graph in [False,True]:
                        key=(model,precision,batch,scope,graph)
                        rows=groups[key]
                        assert sorted(r['round'] for r in rows)==[0,1,2],key
                        wall=np.concatenate([r['wall'] for r in rows])
                        gpu=np.concatenate([r['cuda'] for r in rows])
                        assert len(wall)==len(gpu)==300
                        row=dict(model=model,precision=precision,batch=batch,scope=scope,graph=graph,
                                 samples=300,wall_p50_ms=float(np.percentile(wall,50)),
                                 wall_p95_ms=float(np.percentile(wall,95)),
                                 gpu_p50_ms=float(np.percentile(gpu,50)),gpu_p95_ms=float(np.percentile(gpu,95)),
                                 round_wall_p50_ms=[float(np.median(r['wall'])) for r in sorted(rows,key=lambda x:x['round'])])
                        measurements.append(row)
                        index[key]=row
    assert len(records)==144 and len(measurements)==48
    checks={}
    coverage=[]
    exports={m:json.loads((root/m/'export.json').read_text()) for m in ['dino','b2']}
    for model in ['dino','b2']:
        checks[model]={}
        for precision in ['fp16','int8']:
            checks[model][precision]={}
            for batch in [1,16]:
                checks[model][precision][str(batch)]=json.loads((root/model/f'{precision}_bs{batch}.correctness.json').read_text())
                graphcheck=json.loads((root/model/f'{precision}_bs{batch}.graph_correctness.json').read_text())
                assert max(x['max_abs'] for x in graphcheck.values())<1e-5
        for batch in [1,16]:
            row=dict(model=model,batch=batch)
            for scope in ['backbone','encoder','step']:
                layers=json.loads((root/model/f'{scope}_int8_bs{batch}.layers.json').read_text())['Layers']
                convgemm=[x for x in layers if any(s in x.get('LayerType','').lower() for s in ['conv','gemm','matmul'])]
                ints=[x for x in convgemm if 'int8' in json.dumps(x.get('Inputs',[])+x.get('Outputs',[])).lower() or str(x.get('Weights',{}).get('Type','')).lower()=='int8']
                row[scope]=dict(int8_conv_gemm=len(ints),conv_gemm=len(convgemm))
                assert len(ints)>0, (model,batch,scope,'INT8 did not execute')
            coverage.append(row)
    env=json.loads((root/'runtime_environment.json').read_text())
    builds=[]
    for m in ['dino','b2']:
        for b in [1,16]:
            for p in ['fp16','int8']:
                for scope in ['backbone','encoder','step']:
                    meta=json.loads((root/m/f'{scope}_{p}_bs{b}.build.json').read_text())
                    if p=='int8':
                        assert meta.get('quantization_recipe')=='entropy2_w8a8_static_stride2_fp16_exact_avgpool_v4'
                    builds.append(meta)
            post_meta=json.loads((root/m/f'post_fp32_bs{b}.build.json').read_text())
            post_meta['requested_pipeline_precision']=post_meta['precision']
            post_meta['precision']='fp32'
            builds.append(post_meta)
    scripts=Path(__file__).resolve().parent
    script_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts.iterdir() if p.suffix in ['.py','.sh']}
    result=dict(environment=env,exports=exports,data=json.loads((root/'data_manifest.json').read_text()),
                measurements=measurements,correctness=checks,int8_coverage=coverage,
                builds=builds,script_sha256=script_hashes,
                source_artifacts=str(root.resolve()))
    data_dir=args.report.parent/'data'
    data_dir.mkdir(parents=True,exist_ok=True)
    summary_path=data_dir/'tensorrt_latency_20261029_summary.json'
    summary_path.write_text(json.dumps(result,indent=2))
    (data_dir/'tensorrt_latency_20261029_measurements.jsonl').write_text((root/'measurements.jsonl').read_text())
    labels={'dino':'DINOv3 ConvNeXt-B','b2':'EfficientViT-B2 native'}
    def value(m,p,b,s,g):
        r=index[(m,p,b,s,g)]
        return f"{r['wall_p50_ms']:.2f} / {r['wall_p95_ms']:.2f}"
    lines=['# TensorRT FP16 / INT8 延迟对比（2026-10-29）','',
           'NVIDIA L4 23GB / 72W / 默认动态时钟；TensorRT 10.9.0.34；TF32 关闭。以下为 **p50 / p95，ms/整批**；BS1=3 视角，BS16=48 张 RGB，不除以 16。固定输入，每项 3 轮、每轮预热 30 次后同步计时 100 次；CG=CUDA Graph。','',
           '## 独立主干编码','',
           '输入为预处理后的 RGB（每视角 3×256×320），仅计主干特征提取。','',
           '| 模型 | 精度 | BS | TensorRT | TensorRT + CG |',
           '|---|---|---:|---:|---:|']
    for m in ['dino','b2']:
        for p in ['fp16','int8']:
            for b in [1,16]:
                lines.append(f'| {labels[m]} | {p.upper()} | {b} | {value(m,p,b,"backbone",False)} | {value(m,p,b,"backbone",True)} |')
    lines+=['','## 完整推理管道','',
            'GPU 原始 RGB（240×320）+ state + 固定噪声 → 23×14 动作；包含预处理、主干、融合/压缩、scene adaptor、一次 K/V 计算、FM 采样、反归一化。','',
            '| 模型 | 精度 | BS | 4步 TRT | 4步 TRT+CG | 10步 TRT | 10步 TRT+CG |',
            '|---|---|---:|---:|---:|---:|---:|']
    for m in ['dino','b2']:
        for p in ['fp16','int8']:
            for b in [1,16]:
                lines.append(f'| {labels[m]} | {p.upper()} | {b} | {value(m,p,b,"fm4",False)} | {value(m,p,b,"fm4",True)} | {value(m,p,b,"fm10",False)} | {value(m,p,b,"fm10",True)} |')
    lines+=['','## 量化核验','',
            '显式 Q/DQ PTQ（A8 按张量、W8 按通道，Entropy2）：128 组真实观测（episode 0–7），FM 单步校准 512 样本覆盖 4/10 步全部时刻；BS1/16 共用尺度。32 组未参与校准的观测（episode 48–49）验证，同输入/同噪声对照 FP32；下表取 BS1/16 较差值。','',
            '| 模型 | 精度 | 主干特征 cosine | 4步动作相对L2误差 | 10步动作相对L2误差 |',
            '|---|---|---:|---:|---:|']
    for m in ['dino','b2']:
        for p in ['fp16','int8']:
            cs=list(checks[m][p].values())
            cosine=min(c['tokens']['cosine'] for c in cs)
            errors=[100*max(c[f'action_{s}']['relative_l2'] for c in cs) for s in [4,10]]
            lines.append(f'| {labels[m]} | {p.upper()} | {cosine:.6f} | {errors[0]:.3f}% | {errors[1]:.3f}% |')
    lines+=['','本次 INT8 PTQ 出现明显数值退化，INT8 延迟不代表等精度加速。','',
            'INT8：W8A8，4 个适配器 stride2 卷积保留 FP16，其他非量化算子用 FP16/FP32，后处理 FP32；实际 Conv/GEMM INT8 层数（主干 / 条件编码 / FM 单步）：','',
            '| 模型 | BS1 | BS16 |','|---|---:|---:|']
    for m in ['dino','b2']:
        cols=[]
        for b in [1,16]:
            row=next(r for r in coverage if r['model']==m and r['batch']==b)
            cols.append(' / '.join(f"{row[s]['int8_conv_gemm']}" for s in ['backbone','encoder','step']))
        lines.append(f'| {labels[m]} | {cols[0]} | {cols[1]} |')
    lines+=['',
            '计时不含 H2D/D2H、视频解码、噪声生成、构建/校准/捕获；未测闭环成功率。PyTorch 仅导出与数值校验，计时进程未加载 PyTorch。','',
            'PyTorch：训练/通用推理框架；TensorRT：模型编译优化与低精度推理运行时；CUDA Graph：复用 GPU 执行序列、减少调度开销，可叠加 TensorRT。[NVIDIA 说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)。','',
            '[汇总与权重指纹](data/tensorrt_latency_20261029_summary.json) · [逐次计时](data/tensorrt_latency_20261029_measurements.jsonl) · 复现：`bash tools/quantization/run_trt_latency.sh`。']
    args.report.write_text('\n'.join(lines)+'\n')
    (root/'COMPLETED').write_text('48 matched measurements; 300 samples each; all precision / graph checks complete.\n')
    print(args.report)
    print('REPORT_LINES',len(lines))


if __name__=='__main__':
    main()
