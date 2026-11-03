# B0原始2048 baseline：GR00T式归一化、无池化、FM4

仅本次baseline新训练一次；旧六组为已完成历史对照，不重启或重复评测。

| 配置 | repeat stage4 | 视觉MLP | DiT维度 | 成功数/100 | 来源 |
|---|---|---|---:|---:|---|
| baseline | 否 | Linear128→1024；普通MLP1024→2048→GELU→2048 | 2048 | 51 | 本次全新训练与评测 |
| stage4_resmlp_no_norm | 是 | 残差MLP，无RMSNorm | 1024 | 51 | 此前已完成六组 |
| stage4_rms | 是 | RMSNorm＋残差MLP | 1024 | 55 | 此前已完成六组 |
| mlp1024 | 否 | 普通MLP，Linear128→512 | 1024 | 44 | 此前已完成六组 |
| rms | 否 | RMSNorm＋残差MLP | 1024 | 17 | 此前已完成六组 |
| stage4 | 是 | 普通MLP | 2048 | 43 | 此前已完成六组 |
| stage4_mlp1024 | 是 | 普通MLP，Linear128→512 | 1024 | 46 | 此前已完成六组 |

本次baseline与旧组3（44/100）的差值为+7个百分点。每组只有一个训练seed，不能当作多训练seed均值。

共同beat_block_hammer/demo_clean、50demo（49训练/1留出）、三视角、seed42、200epoch8771步、batch128、BF16；groot=q01/q99分位数映射并裁剪[-1,1]、常量维归零，state/action独立统计、预测逆变换也裁剪；tau_token、标准FM1000桶、推理4steps[0,250,500,750]。闭环最终main、eager/BF16/KVon/GraphOff/RGB不翻，100相同固定唯一seed与100非空视频。

本次网络：原预训练冻结B0输出80tokens/view→原128维多视图融合→原两个可训练ResBlock降至2×3→不池化→Linear128→1024→展平三视图18token→普通MLP1024→2048→GELU→2048。无RMSNorm视觉残差、repeat stage4、桥接或独立MBConv；DiT/state/动作/t2048，8层32heads/head_dim64，DiT FFN固定2048→2048→GELU→2048。

已对比旧组3解析配置：除宽度1024→2048、heads16→32、视觉Linear输出512→1024与variant名称外，其余训练/数据/归一化/池化/推理一致；974源码与旧六组相同，未修改网络实现。config_comparison.json保存逐字段差异。

本次总参数361,220,992，可训练360,537,998。CPU真实数据预检、八卡两次forward/backward零optimizer启动检查、初始和最终main/EMA严格重载、18token/KV/实际四次FM、2000→4000原ResBlock/MLP/融合/Linear/DiT FFN更新及186个冻结B0张量匹配均通过。冻结commit `d8833058e3e831d68b5b0d176865823c4d7a3aaf`。

W&B [finished / 8771](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/cd61yq3l)，独立API核验8771步；最终训练loss 0.00123。

原runner同seed场景恢复4次，保留原始日志，未替换seed或因policy失败重跑挑结果。

训练：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_groot_nopool_baseline_fm4_20261102_v1/baseline`
闭环：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_groot_nopool_baseline_fm4_20261102_v1/baseline/main_nfe4_100_v1/result.json`
完整审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_groot_nopool_baseline_fm4_20261102_v1/final_audit.json`
