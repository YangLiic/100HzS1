# handover_block：16组实验（进行中）

截至2026-10-31 14:18 CST：B0/Base的8组训练均已完成并通过主权重与EMA严格重载，其中7组100种子闭环已核验；Base MeanFlow AdaRMS（300epoch）的1-step闭环进行中。B1/B2八组继续等待第一波全部闭环核验完成。当前最高为B0 FM tau_AdaRMS的22/100。未完成项不计为0成功率。

每组使用同一100个专家验证种子；PyTorch eager BF16、KV开启、CUDA Graph关闭、RGB不翻转、绝对joint14动作、JPEG roundtrip。FM4-step，MeanFlow1-step。评测使用主权重；主权重与EMA都严格重载，但这不代表EMA数值质量已验证正常，相关精度问题见DINO审计报告。

| Backbone | 算法 | tau注入 | Epoch | 推理步数 | 状态 | 成功次数/成功率 |
|---|---|---|---:|---:|---|---:|
| B0 | FM | tau_token | 200 | 4 | 训练、100种子闭环已核验 | 1/100（1%） |
| B0 | FM | tau_AdaRMS | 200 | 4 | 训练、100种子闭环已核验 | 22/100（22%） |
| B0 | MeanFlow | tau_token | 200 | 1 | 训练、100种子闭环已核验 | 5/100（5%） |
| B0 | MeanFlow | tau_AdaRMS | 300 | 1 | 训练、100种子闭环已核验 | 1/100（1%） |
| DINO ConvNeXt-Base | FM | tau_token | 200 | 4 | 训练、100种子闭环已核验 | 3/100（3%） |
| DINO ConvNeXt-Base | FM | tau_AdaRMS | 200 | 4 | 训练、100种子闭环已核验 | 9/100（9%） |
| DINO ConvNeXt-Base | MeanFlow | tau_token | 300 | 1 | 训练、100种子闭环已核验 | 0/100（0%） |
| DINO ConvNeXt-Base | MeanFlow | tau_AdaRMS | 300 | 1 | 训练已完成，100种子闭环进行中 | — |
| B1 | FM | tau_token | 300（计划） | 4 | 等待第一波完成 | — |
| B1 | FM | tau_AdaRMS | 300（计划） | 4 | 等待第一波完成 | — |
| B1 | MeanFlow | tau_token | 300（计划） | 1 | 等待第一波完成 | — |
| B1 | MeanFlow | tau_AdaRMS | 300（计划） | 1 | 等待第一波完成 | — |
| B2 | FM | tau_token | 300（计划） | 4 | 等待第一波完成 | — |
| B2 | FM | tau_AdaRMS | 300（计划） | 4 | 等待第一波完成 | — |
| B2 | MeanFlow | tau_token | 300（计划） | 1 | 等待第一波完成 | — |
| B2 | MeanFlow | tau_AdaRMS | 300（计划） | 1 | 等待第一波完成 | — |

## 训练预算与数据

50条demo中实际49条用于训练，episode4留出验证；全局batch128、冻结视觉backbone、训练视频缓存开启、BGR→RGB、训练不用推理KV缓存。纯MeanFlow连续t/r、equal_time_ratio=0、time_embed_scale=1、guidance_scale=1、无Dispersive。

原先已开始的5组保持200epoch，其余11组按用户要求改为300epoch。已开始项目的Epoch来自各自resolved_config.yaml；未开始的“300（计划）”需训练后从实际解析配置再次核验。200/300训练预算不同，不能作为相同训练时长的公平比较。B1/B2使用独立300epoch队列，在B0/Base八组训练及100种子闭环都核验后才开始。

## 证据与产物

- checkpoint：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/handover_block/handover_block_tau_matrix_20261030_v1/<backbone>_<flow>_<injection>/`
- 逐种子结果：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/handover_block_tau_matrix_20261030_v1/<backbone>_<flow>_<injection>_nfe<N>/result.json`
- 各已完成项重新计数成功次数，核验100个唯一seed与注册表完全一致，并核查真实后端、精度、步数、KV/Graph、颜色和checkpoint。
- 手交任务种子SHA256：`43f1a8201f9f558bd5644327e6154d72b29b78650162104b7d4a0f79ea72e0cb`。
- W&B在训练期间由独立CPU relay持续同步；第一波8组均有完成记录，最新服务端核验Base MeanFlow AdaRMS为finished/32453，其余7组此前已核验。已完成step按各run实际21635或32453区分；新训练开始后继续检查其服务端同步。
- DINO实现排查：同目录`dino_backbone_pipeline_audit_20261031.md`。其中local machine的beat NFE20诊断与本表独立，不计入这16组，也不修改本表推理口径。
- 初始专家种子构建中对特定missing-target断言使用独立guard进行重试/拒绝；最终注册表已固定100合法种子，未在模型间更换种子。正式训练源码和运行任务未为此改写。

这是阶段表，不是16组全部完成的最终报告。
