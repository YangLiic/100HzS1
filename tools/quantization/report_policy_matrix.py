"""CPU-only final audit and Markdown for the four FP16 policy measurements."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True)
    args=ap.parse_args();p=read(args.plan);root=Path(p['root']);art=Path(p['artifacts'])
    for relative,digest in p['source_hashes'].items():
        assert sha(Path(p['snapshot'])/relative)==digest,relative
    result=[]
    for case in p['cases']:
        c=art/case['id'];exp=read(c/'export_manifest.json');built=read(c/'build_manifest.json')
        par=read(c/'parity.json');lat=read(c/'latency.json')
        assert par['all_finite'] and par['action_basic_tolerance']
        assert len(par['results'])==len(p['reference_indices'])==4
        assert lat['nfe']==4 and lat['actual_time_buckets']==[0,250,500,750]
        assert lat['views']==3 and lat['batch']==1 and lat['kv_cache'] and not lat['cuda_graph']
        assert exp['conditions']==(12 if case['pooling'] else 18) and exp['fusion_ln_count']==5
        assert lat['engine_hashes']==built['engines']
        for rel,digest in built['engines'].items():assert sha(c/rel)==digest
        for name in ['model.safetensors','config.json']:
            expected=exp['weights_sha256' if name.startswith('model.') else 'config_sha256']
            assert sha(Path(case['checkpoint'])/name)==expected
        assert all(x['samples']==p['repeats']*p['rounds'] for x in lat['scopes'].values())
        result.append(dict(case=case,latency=lat,parity=par,export=exp))
    assert len({r['latency']['gpu'].split(',')[1].strip() for r in result})==1
    diag=read(root/'diagnosis.json');pt=read(root/'pytorch_precision_diagnosis.json')
    text=['# B0 baseline与共享stage4＋RMS残差MLP：FP16分项推理时间','',
          '## 结论口径','',
          '5.2%是编码器特征相对FP32的RMSE，并非关节动作误差5.2、更不是成功率下降5.2个百分点。此前5%是探针自设检查线，不是行业标准。',
          '同一ONNX在TensorRT FP32下与PyTorch FP32相对误差约0.0023%；PyTorch自身autocast FP16约5.98%；旧TensorRT FP16约5.19%。这表明原导出数学路径基本正确，特征存在FP16敏感性，不能归因为policy代码计算错误。',
          '原EfficientViT LiteMLA将attention显式转float；新构建器保留其敏感attention、归一化和softmax计算为FP32，卷积/Linear等使用TensorRT FP16。保护后的单编码器相对误差约4.59%。这属于FP16部署的混合精度策略，未改网络、权重或训练。',
          '以下四组使用真实已训练权重，分别做同输入/state/噪声的最终动作数值对照后测速；这些对照不等同于TensorRT闭环成功率验证。',
          '', '## GPU输入下的整批P50（ms）','',
          '| policy | 池化 | 保存归一化 | 视觉token | DiT维度 | 完整FM4 | 视觉编码含融合/MLP | B0（新网络含repeat） | baseline原压缩模块 | DiT纯前向×4 |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result:
        c=r['case'];s=r['latency']['scopes'];base=c['variant']=='baseline'
        text.append('| {} | {} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {} | {:.4f} |'.format(
            'baseline' if base else 'stage4＋RMS', '2×2' if c['pooling'] else '无', c['norm'],
            12 if c['pooling'] else 18,c['dimension'],s['full_fm4']['gpu_p50_ms'],s['visual']['gpu_p50_ms'],
            s['backbone']['gpu_p50_ms'],format(s['adapter']['gpu_p50_ms'],'.4f') if base else '不适用',s['dit_four_calls']['gpu_p50_ms']))
    text+=['', 'baseline原压缩模块计时包括两个原ResBlock、可选2×2池化与Linear128→1024。新网络没有这两个融合后ResBlock，其token压缩已经计入“B0含repeat”；仍保留可选池化与Linear，适配器独立计时在原始JSON中。不要把新网络的repeat再重复算一次。',
           '视觉编码包括raw RGB预处理、B0/共享stage4、三视角融合、空间适配器/池化/Linear、视觉普通或RMS残差MLP，不含KV生成。完整FM4额外包含KV准备、状态/动作MLP、四次DiT、Euler更新和后处理；DiT纯前向×4使用真实四步轨迹对应的已嵌入输入，不含状态MLP、KV生成或Euler。',
           '各子模块单独编译和计时，TensorRT融合边界与完整pipeline不同，因此分项不能直接相加代替完整推理时间。',
           '', '## 同步wall与I/O口径','',
           '| case | 完整FM4 GPU P95 ms | GPU输入完整FM4 wall P50 ms | 含CPU输入H2D/输出D2H的service wall P50 ms | service wall P95 ms |',
           '|---|---:|---:|---:|---:|']
    for r in result:
        s=r['latency'];f=s['scopes']['full_fm4']
        text.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(r['case']['id'],f['gpu_p95_ms'],f['wall_p50_ms'],s['service_wall_p50_ms'],s['service_wall_p95_ms']))
    text+=['', 'service时间包含Python输入校验、RGB/state/既定noise上传与动作下载；不含相机采集、JPEG编码/解码、网络通信或随机噪声生成。GPU输入表排除了H2D/D2H，输入提前驻留GPU。',
           '', '## 最终动作数值对照','',
           '| case | 四个真实输入中最大动作RMSE | 最大动作绝对误差 | 原框架基础检查线 |',
           '|---|---:|---:|---|']
    for r in result:
        rows=r['parity']['results']
        text.append('| {} | {:.7f} | {:.7f} | RMSE≤0.01且max≤0.05，通过 |'.format(r['case']['id'],max(x['action']['rmse'] for x in rows),max(x['action']['max_abs'] for x in rows)))
    text+=['', '该基础检查线用于捕获明显导出偏差，并非用户任务的成功率验收线；未进行新的RoboTwin闭环评测。所有中间特征和最终动作误差均保留在parity.json中。',
           '', '## 测试条件与来源','',
           '统一三视角、bs=1、标准FM、tau_token、1000时间桶、4次[0,250,500,750]前向、KV开启、CUDA Graph关闭、TensorRT FP16主计算与FP32敏感计算/后处理。raw RGB240×320进入预处理后256×320。',
           '四组均在同一个L4/GPU UUID上计时：`'+result[0]['latency']['gpu']+'`；TensorRT '+result[0]['latency']['tensorrt']+'。build热队列使用其他GPU，避免与测速卡争用；每项预热{}次、每轮{}次、共{}轮。计时进程未导入PyTorch。'.format(p['warmup'],p['repeats'],p['rounds']),
           '池化组采用以前已训练的minmax权重，不池化组采用已训练的groot权重；同一池化设置下baseline/新网络归一化一致。跨池化比较同时存在权重和normalizer差异，不能称为固定同一权重只改池化开关的因果实验。本任务只比较实际网络推理成本，未重训。',
           '精度参考：[NVIDIA Accuracy Considerations](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/accuracy-considerations.html)。',
           '', '## 产物路径','', '全部引擎、ONNX、refs、构建详情、数值对照、原始计时：`'+str(art)+'`',
           '封存源码、request、队列、日志与审计：`'+str(root)+'`']
    for r in result:text+=['',r['case']['id']+'：`'+r['case']['checkpoint']+'`']
    text+=['','源码基于commit `'+p['commit']+'`并按source_hashes封存新增量化工具；未push。','']
    Path(p['report']).write_text('\n'.join(text))
    audit=dict(status='verified',cases=result,diagnosis=diag,pytorch_precision=pt,
               report=p['report'],source_files=len(p['source_hashes']))
    write=root/'final_audit.json';write.write_text(json.dumps(audit,indent=2))
    print('FINAL_REPORT',p['report'],flush=True)


if __name__=='__main__':main()
