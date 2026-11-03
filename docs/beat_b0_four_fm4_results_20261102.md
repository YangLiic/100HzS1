# B0 四组标准 FM4 实验结果

四种结构均从同一 B0 预训练重新训练一次，使用同一固定100 seed闭环；没有复用历史策略权重。

| 配置 | 机器 | DiT宽度 | 成功/100 | 最终训练loss | W&B |
|---|---|---:|---:|---:|---|
| baseline | machine1 | 2048 | 52/100 | 0.00115 | [FINISHED · 8771步](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/7pvnlqx1) |
| stage4 | machine2 | 2048 | 32/100 | 0.00306 | [FINISHED · 8771步](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/28x6jos8) |
| rms | machine1 | 1024 | 15/100 | 0.00167 | [FINISHED · 8771步](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/vftvo1q2) |
| stage4_rms | machine2 | 1024 | 45/100 | 0.00261 | [FINISHED · 8771步](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/5tvf6pkq) |

共同：beat_block_hammer/demo_clean，50demo（49训练/1留出），seed42、200epoch/8771步、batch128、三视角、minmax、BF16，标准FM/tau_token/1000时间桶，4step/eager/KV开启/Graph关闭/闭环RGB不翻转。

所有组融合后池化2×2、Linear128→1024，最终12个视觉token。stage4组融合前共享冻结stage4额外两次和一个可训练128→64桥接；RMS组只在视觉侧使用Pre-RMS残差MLP。DiT实现未改，FFN始终D→D→GELU→D。

冻结提交：`1c254e5c2fb754c4b872727b57d5a82f57d691a2`；核验971份源码/配置文件。
每组main和EMA最终严格重载通过、真实FM时间桶[0,250,500,750]；2000→4000模块更新和冻结B0逐张量匹配通过。每组100唯一seed及100非空视频核验通过。启动检查无optimizer step，正式训练各一次。

本次每组只有一次训练，100回合成功率不能代表多训练seed的稳定均值。

## 产物路径

- **baseline**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_four_variants_fm4_20261102_v1/baseline`；评测 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_four_variants_fm4_20261102_v1/baseline/main_nfe4_100_v1/result.json`。
- **stage4**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_four_variants_fm4_20261102_v1/stage4`；评测 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_four_variants_fm4_20261102_v1/stage4/main_nfe4_100_v1/result.json`。
- **rms**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_four_variants_fm4_20261102_v1/rms`；评测 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_four_variants_fm4_20261102_v1/rms/main_nfe4_100_v1/result.json`。
- **stage4_rms**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_four_variants_fm4_20261102_v1/stage4_rms`；评测 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_four_variants_fm4_20261102_v1/stage4_rms/main_nfe4_100_v1/result.json`。

完整审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_four_variants_fm4_20261102_v1/final_audit.json`。

## 补充对比

| 配置 | 成功数/100 | 相对 baseline | 全模型参数 M | 可训练参数 M | 同 seed 场景自动重试次数 |
|---|---:|---:|---:|---:|---:|
| baseline | 52 | +0 个百分点 | 361.209 | 360.526 | 2 |
| stage4 | 32 | -20 个百分点 | 360.593 | 359.910 | 7 |
| rms | 15 | -37 个百分点 | 94.428 | 93.745 | 3 |
| stage4_rms | 45 | -7 个百分点 | 93.812 | 93.129 | 4 |

参数统计覆盖整套策略（包括视觉编码器、融合/压缩、视觉MLP、状态/动作/时间与DiT）；共享stage4只计一次。池化无参数，stage4压缩增加的共享桥接仅8192参数，并移除原两个ResBlock。不能把整个模型的参数下降都归因于视觉MLP。

本轮 baseline 为 52/100；单独 stage4 为 32/100，单独 RMS 降维为 15/100，联合方案为 45/100。以本次计数为准，尚不能宣称降低至1024维保持成功率无损。两个消融因素的组合表现应独立看待，不把单独分支的变化简单相加。

评测中出现了间歇性的专家场景检查失败。原评测器按同一固定seed自动重启相应worker；没有替换seed、没有筛掉失败回合，也没有手工重复评测挑结果。全部请求最终成功，四组各100个唯一seed和100个非空视频齐全。表中重试次数来自node日志的WORKER_SCENE_RETRY事件，不是新增评测回合数；完整事件清单见运行目录report_supplement.json。

每组只有一个训练seed（42）和一次训练，固定100回合的差异不等于多次训练的稳定平均差异。本轮没有追加其他实验、EMA评测或NFE扫描。
