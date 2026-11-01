# B0 RMSNorm残差1024／FFN2048：最终结果

> 历史实验：此报告对应已完成的16%宽FFN模型。用户随后要求恢复固定等宽DiT FFN，当前开发代码已移除独立宽度配置。
> 原权重、结果、配置和冻结源码仍保留；复现本报告须使用其3994192快照，不能严格加载到当前等宽DiT。

最终main闭环 **16/100（16%）**，四分片成功数4＋5＋2＋5。100个唯一固定seed、100个独立非空视频及完整最终审计均通过。
北京时间2026年11月1日23:24完成request，machine1已恢复holding。只执行了一次8771步正式训练。

| 方案 | DiT宽度 | 成功数/100 |
|---|---:|---:|
| 原始B0 baseline | 2048 | 52 |
| 原直接降维128→512→1024→1024 | 1024 | 22 |
| 本次RMSNorm残差＋FFN加宽 | 1024 | **16** |

本次未达到降低DiT宽度同时保持成功率的目标。它是视觉投影保留1024、scene残差RMS及DiT FFN加宽的联合方案，不能据此单独归因于RMSNorm；单训练seed和100回合也不能证明16与22的差异稳定存在。

## 实际网络与训练

原B0预训练冻结，三视角各128×8×10先融合，再经原两个可训练ResBlock和原AdaptiveAvgPool2d(2,2)。
Linear128→1024后展平为12个token，scene为u＋Linear2048→1024(GELU(Linear1024→2048(RMSNorm(u))))。
DiT宽1024、8层16heads、head_dim64，各层FFN1024→2048→GELU→1024；KV形状[1,16,12,64]。
状态／带噪动作共享原三Linear架构14→1024→1024→1024，t宽1024，无MeanFlow r分支。
无stage4额外重复、无独立MBConv、无教师或分阶段训练。

总参数111,213,324；可训练110,530,446。
beat_block_hammer/demo_clean，50demo（49训练1验证，5614/118样本）、seed42、200epoch／8771步、batch128、BF16、minmax、三视角left/right/head、绝对joint14、horizon24。
标准FM／tau_token／1000时间桶／4步推理；main与EMA实测DiT调用[0,250,500,750]。
最终loss0.00115，日志训练耗时约16分钟；无非有限值、OOM或运行错误。

八卡启动检查仅两次前向反向，optimizer_steps=0，全部参数和缓冲不变。
v1只在启动检查DDP设备顺序处失败，未执行正式训练；唯一正式训练在独立v2 request执行。

## 核验与评测

最终main和EMA均严格重载成功，冻结B0按各自保存精度匹配预训练。
2000→4000 checkpoint证明scene RMS gamma／FC1／FC2、DiT FFN及视觉Linear发生更新；EMA FCs也更新。
每个检查点的main/EMA均有186个B0张量保持不变。EMA gamma在BF16下未表现出量化后的变化，已在审计中记录；评测使用main。
977个冻结源码、封存配置、预训练、最终权重及seed哈希核验通过。

闭环最终main，同machine1，两条独立4GPU lane、四个25-seed分片；FM NFE4／eager／BF16／KV-on／Graph-off。
离线BGR转RGB；SAPIEN闭环RGB不翻转。全部100seed与既有baseline/降维实验相同，未替换seed。
seed SHA256：fca2035b8f623d52f980843f8f578710c520997db5c7f334f557ebec5a2ec8da。

## 权重、结果与日志

- 训练权重：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_fm4_resmlp_d1024_ffn2048_20261101_v2`
- 评测汇总：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_fm4_resmlp_d1024_ffn2048_20261101_v2/main_nfe4_100_v1/result.json`
- 最终审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_fm4_resmlp_d1024_ffn2048_20261101_v2/final_audit.json`
- 冻结源码：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_fm4_resmlp_d1024_ffn2048_20261101_v2/snapshot`，commit `399419213ab0f928c424ef00443c41bb1e121d6a`
- 配置：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_fm4_resmlp_d1024_ffn2048_20261101_v2/configs/train.yaml`
- request：`s1-beat-b0-fm4-resmlp1024-20261101-v2`，Job `t-20260820125318-wd2mn`
- W&B：[正式run](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/9c7phjk7)，实际服务端 **FINISHED／8771**。

所有本次权重、100个视频、日志和快照保留。此前用户授权删除的融合后stage4失败组属于另一组，删除前后本次保护文件哈希未变。
新准备的“融合前共享stage4＋0参数2×2池化”只有代码与独立YAML，尚未提交训练或评测。
开发HEAD中删除旧融合后分支的改动也通过当前权重main/EMA严格重载；本次运行仍采用冻结3994192源码。
