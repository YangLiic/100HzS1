# B0 标准 FM4：降维与共享 stage4 压缩结果

两组独立训练及各自100固定seed闭环已完成，完整最终审计通过。
共同设置为标准FM、tau_token、4-step、三视角、状态/动作min-max，
beat_block_hammer/demo_clean，50demo、seed42、200epoch8771步、BF16，
eager/KV-on/Graph-off，闭环RGB不翻转。只评测最终main，不评测EMA。

| 实验 | 视觉压缩与维度 | 最终成功数 | 最终训练loss | 训练耗时 |
|---|---|---|---|---|
| machine1 width1024 | 原融合后每视角2×2；Linear128→512，sceneMLP512→1024→GELU→1024；三视角12token，DiT/state/action/t1024 | 22/100（22%） | 0.00274 | 14:33 |
| machine2 shared-stage4 | 同一个冻结stage4额外调用两次，共享可训练128→64桥接，80→20→6token/视角后融合；Linear128→1024，sceneMLP1024→2048→GELU→2048；18token，DiT/state/action/t2048 | 48/100（48%） | 0.00204 | 37:18 |

共享 stage4 包含原stride2入口MBConv及后续两个EfficientViTBlock，
各block包含注意力与局部MBConv；复用整个冻结预训练模块。
这不是新增两层独立MBConv。两次额外调用共用同一个8192参数的1×1桥接。
CPU梯度检查和正式checkpoint2000→4000对比均确认桥接更新；
主模型8192/8192桥接元素变化，EMA5886/8192变化，预训练B0的186个张量
在main/EMA分别按保存精度仍精确匹配。最终主模型与EMA严格重载也通过。

两组真实采样均核验4次DiT前向，时间桶[0,250,500,750]，只有t嵌入、无r；
有限action[1,23,14]，所有权重有限。封存源码973文件和原始870实现文件、
训练配置、预训练、seed及最终模型SHA全部核验；没有改动运行或冻结源码。
每组4分片各25个，100个唯一seed与原计划完全一致，100个独立非空视频存在。

| 分片 | width1024成功 | shared-stage4成功 |
|---|---|---|
| shard0 | 7/25 | 11/25 |
| shard1 | 5/25 | 10/25 |
| shard2 | 5/25 | 14/25 |
| shard3 | 5/25 | 13/25 |

两个唯一request均成功退出exit0并恢复holding：width于19:56:46、stage4于20:19:18
（2026-11-01，Asia/Shanghai）。两组W&B均独立确认服务端finished、8771步，
最近指标有限，日志转发完整。未追加EMA评测、NFE扫描或其他实验，未push。

这两行同时具有不同的宽度和压缩方式，不能用它们之间的差值单独归因于
宽度或压缩。用户已另外明确授权原始B0、不降维、不额外stage4压缩的FM4
baseline；其独立新run为beat_b0_fm4_native_d2048_baseline_20261101_v1。
待该baseline完成后，再按相同100seed口径比较。旧MeanFlow实验不充当FM基线。

## width1024_post2x2 产物

- 权重：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_fm4_ablation_20261101_v1/width1024_post2x2`。
- 100回合汇总：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_fm4_ablation_20261101_v1/width1024_post2x2/main_nfe4_100_v1/result.json`。
- 最终审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_fm4_ablation_20261101_v1/cases/width1024_post2x2/final_audit.json`。
- W&B：[正式训练](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/pt5tbit0)，finished8771。
- mainSHA256：`4f9c2cb5bd690c218c0a40d9a13374504b6affab16c1c334690baeea3a6616a3`。
- EMA SHA256：`e128718e07d164374048d268f727aec7787c90b6f3b64a529f39cd1eadc2eb6c`。

## stage4_d2048 产物

- 权重：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_fm4_ablation_20261101_v1/stage4_d2048`。
- 100回合汇总：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_fm4_ablation_20261101_v1/stage4_d2048/main_nfe4_100_v1/result.json`。
- 最终审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_fm4_ablation_20261101_v1/cases/stage4_d2048/final_audit.json`。
- W&B：[正式训练](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/bnjepzv2)，finished8771。
- mainSHA256：`6c3a15cce99643d9259bc142b6f032ce36950270a578c5485610ca828e4b5d7c`。
- EMA SHA256：`df2c81b8717690256a4fb49c9c35d926adac7a254964dbbb9e0d0ac30a2be0e0`。
