# handover_block：已完成训练与闭环评测结果

整理时间：2026-10-31 15:48 CST。原16组实验计划已按用户要求停止；本文仅汇总已完成的8组训练及各自100种子闭环。B1/B2的未完成实验不计入结果。

**当前最高成功率：EfficientViT B0 + FM（4-step）+ tau_AdaRMS，22/100（22%），训练200 epoch。**

| Backbone | 方法 | tau注入 | Epoch | 优化步数 | 推理步数 | 成功次数 | 成功率 |
|---|---|---|---:|---:|---:|---:|---:|
| EfficientViT B0 | FM | tau_token | 200 | 21635 | 4 | 1/100 | 1% |
| EfficientViT B0 | FM | tau_AdaRMS | 200 | 21635 | 4 | 22/100 | 22% |
| EfficientViT B0 | MeanFlow | tau_token | 200 | 21635 | 1 | 5/100 | 5% |
| EfficientViT B0 | MeanFlow | tau_AdaRMS | 300 | 32453 | 1 | 1/100 | 1% |
| DINOv3 ConvNeXt-Base | FM | tau_token | 200 | 21635 | 4 | 3/100 | 3% |
| DINOv3 ConvNeXt-Base | FM | tau_AdaRMS | 200 | 21635 | 4 | 9/100 | 9% |
| DINOv3 ConvNeXt-Base | MeanFlow | tau_token | 300 | 32453 | 1 | 0/100 | 0% |
| DINOv3 ConvNeXt-Base | MeanFlow | tau_AdaRMS | 300 | 32453 | 1 | 6/100 | 6% |

## 数据与训练口径

- 50条demo，49条用于训练、episode4留出验证；留出验证集不是独立测试集。
- 全局batch=128，冻结预训练视觉backbone；训练开启视频缓存和BGR→RGB，训练不用推理KV缓存。
- 绝对joint14关节空间，state观测、action监督；horizon=24，包含1个state和23个未来动作。
- 纯MeanFlow连续t/r，equal_time_ratio=0、time_embed_scale=1、guidance_scale=1、无Dispersive。
- 用户调整训练时长后，已开始的5组保持200epoch，其余已完成的3组为300epoch。表中epoch来自各自resolved_config.yaml，优化步数来自各训练日志；200/300训练预算不同，比较结果时需同时考虑。

## 闭环评测与核验

- 每组使用相同100个任务专属、专家验证种子；成功率分母均为完整100个唯一seed，没有使用未完成结果或更换种子。
- 主权重、PyTorch/eager BF16；KV缓存开启、CUDA Graph关闭；SAPIEN RGB不翻转，JPEG roundtrip开启。FM使用4-step，MeanFlow使用1-step。
- 已交叉核验result.json、原始worker journal及实际推理日志：100条唯一episode、成功次数、真实backend/NFE、23动作chunk、颜色、KV/Graph、冻结源码与checkpoint配置一致。
- 8组主权重和EMA均通过严格重载；本表成功率均来自主权重。EMA数值精度排查另见同目录`dino_backbone_pipeline_audit_20261031.md`。
- W&B的8个正式训练run均已核验服务端finished，最终step与表中各自优化步数一致。
- seed列表SHA256：`43f1a8201f9f558bd5644327e6154d72b29b78650162104b7d4a0f79ea72e0cb`。初始种子筛选针对特定missing-target断言使用独立guard重试/拒绝，最终固定同一100合法种子，未在模型间更换。

## 保留结果与停止清理

- 已保留以上8组完成权重（含EMA）、训练配置、日志及完整闭环结果；权重文件元数据与8个result.json哈希经清理前后核对一致。
- machine1/2上的两个未完成B1 FM训练已取消；其未完成训练目录、本地同步文件和两个未完成W&B run已删除，6个尚未开始的训练已撤出待执行队列。
- 用户停止时8组闭环均已完成，没有仍在运行的评测。待执行/运行训练与评测队列均为空；CPU同步管理器和自动跟进已停止，machine1/2恢复保卡状态。
- 完成权重：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/handover_block/handover_block_tau_matrix_20261030_v1/<backbone>_<flow>_<injection>/`。
- 逐种子结果：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/handover_block_tau_matrix_20261030_v1/<backbone>_<flow>_<injection>_nfe<N>/result.json`。
- 完整8组核验证据：`/DRoboticsAILab/users/licheng.yang/100HzS1_tmp/handover_block_tau_matrix_20261030_v1/firstwave_complete_gate_audit_20261031.json`。
- 停止与清理记录：`/DRoboticsAILab/users/licheng.yang/100HzS1_tmp/handover_block_tau_matrix_20261030_v1/stopped_by_user_cleanup_20261031.json`。
