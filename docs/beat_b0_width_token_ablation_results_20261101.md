# Beat B0 独立消融实验结果

2026-11-01，machine2，源码提交 `bf45128742cc0f08ed5c769e5e0a348246cb54be`。两组均已完成训练和闭环评测，全部最终核验通过。

| 实验 | 压缩方式 | 视觉投影及 scene MLP | DiT 隐藏宽度 | 最终训练 loss | 闭环成功率 |
|---|---|---|---:|---:|---:|
| A：仅降维 | 保留融合后 ResNet 压缩和自适应池化，2×2／视图 | Linear 128→256；MLP 256→512→GELU→512 | 512 | 0.07238 | **0/100（0%）** |
| B：仅改压缩机制 | ViT 内共享 stage4 额外重复两次，自然 2×3／视图，再融合 | Linear 128→1024；MLP 1024→2048→GELU→2048 | 2048 | 0.03309 | **35/100（35%）** |

B 的 stage4 输入64维、输出128维，因此加入一个共享、可训练的1×1桥接层128→64（8192参数、无bias）。两次额外调用复用同一桥接层和同一份原 stage4；原预训练主干及 BatchNorm 冻结，不再使用原 ResNet 压缩模块或自适应池化。

两组状态／动作 token 和 t/r 时间嵌入分别与512、2048维对齐，state adaptor保留三层MLP。物理关节状态和动作仍为14维。A视觉条件为12×512，B为18×2048。

共同设置：beat_block_hammer/demo_clean，B0预训练主干，50条demo（49训练／1留出验证）、seed42、minmax、绝对关节动作、三视图、horizon24、MeanFlow/tau_token、BF16、200epochs、global batch128、AdamW lr1e-4。每组8771步。训练数据BGR→RGB，闭环SAPIEN RGB不翻转。

闭环均使用最终MAIN权重、PyTorch/eager BF16、NFE1、KV开启、CUDA Graph关闭，同一套100个固定专家验证seed；每组均保留100个完整视频。种子文件SHA256：`fca2035b8f623d52f980843f8f578710c520997db5c7f334f557ebec5a2ec8da`。

这些结果展示了两种设置各自的表现；单凭A和B互相比，不能把差异完全归因于维度或压缩机制，因为两组的宽度和压缩方式均不同。要判断各自相对原版的变化，需对照同配方的原始B0/2×2/2048基线。本次未新增第三组训练。

## 产物位置

训练权重共同前缀：

```text
/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_width_token_ablation_20261101_v1/
```

其下分别为 `A_width512_post2x2` 和 `B_stage4_d2048`，每个目录含最终主权重、EMA、配置、训练日志及严格重载报告。

闭环结果共同前缀：

```text
/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_width_token_ablation_20261101_v1/
```

其下为 `<case>/main_nfe1_100_v1/result.json`；分片目录 `shard0`～`shard3` 内有逐回合记录、日志和 `videos/`。

冻结源码、配置、调度器、分片队列、最终审计：

```text
/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_width_token_ablation_20261101_v1/
```

总request：`s1-beat-width-token-ablation-20261101-v1`，2026-11-01 15:43:11（Asia/Shanghai）成功退出，machine2已自动恢复保卡。

## W&B 与核验

- A：[v1s5ar5j](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/v1s5ar5j)
- B：[ppkjetqq](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/ppkjetqq)

服务端独立核验于15:45完成：两组状态均为finished，lastHistoryStep和summary_step均为8771。初始CPU日志转发空文件问题已修复，正式日志已完整同步。

主权重与EMA严格重载及有限动作输出[1,23,14]均通过；最终权重、配置、冻结源码和调度脚本哈希一致；每组四分片各25回合、100唯一seed完全覆盖、100个视频非空，汇总成功数已从逐回合结果重新计数。未替换seed，未覆盖旧产物，未push。
