# B0 共享stage4：关闭池化实验结果

两组使用model.visual_pooling=false重新训练，保留自然2×3、每视角6个token，三视角18token；对照为此前独立完成的pool=true结果，没有重跑对照。

| 配置 | DiT宽度 | 无池化成功/100 | 原2×2池化成功/100 | 差值 | 最终训练loss | W&B |
|---|---:|---:|---:|---:|---:|---|
| stage4 | 2048 | 50/100 | 32/100 | +18个百分点 | 0.00257 | [FINISHED · 8771步](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/h6255ak7) |
| stage4_rms | 1024 | 57/100 | 45/100 | +12个百分点 | 0.00308 | [FINISHED · 8771步](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/tso2flyl) |

共同：beat_block_hammer/demo_clean、50demo（49训练/1留出）、seed42、200epoch8771步、batch128、3视角、minmax、BF16、标准FM/tau_token/1000桶；闭环4step/eager/KV开启/Graph关闭/RGB不翻转。

本次与对应池化组的解析配置逐项比对，网络配置仅pooling开关及派生网格2×2→2×3变化，其他模型和训练/数据/推理设置保持。原冻结B0后共享同一stage4额外两次、一个可训练128→64桥接，6token/视角先融合，再直接Linear128→1024，完全跳过平均池化。

stage4的sceneMLP1024→2048→2048、D2048；stage4_rms的视觉Pre-RMS残差MLP1024→2048→1024、D1024。DiT实现未改，8层、head_dim64、FFN固定D→D→GELU→D。各层KV长度18。

冻结提交：`9b6d6499d6cacfc5927acb903e7180c6d14e4c9b`；核验971份源码/配置文件。
每组初始和最终main/EMA严格重载通过、真实FM采样[0,250,500,750]，2000→4000桥接/RMS/融合/投影/FFN真实更新及原B0全部186张量冻结匹配通过。每组100唯一固定seed和100非空唯一视频齐全。启动检查零optimizer，正式训练各一次。

仅一个训练seed42、每组一次训练，100回合计数不能保证多次训练平均性能或无损。没有额外EMA评测/NFE扫描。

## 产物路径

- **stage4**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_stage4_nopool_fm4_20261102_v1/stage4`；评测 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_stage4_nopool_fm4_20261102_v1/stage4/main_nfe4_100_v1/result.json`。
- **stage4_rms**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_stage4_nopool_fm4_20261102_v1/stage4_rms`；评测 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_stage4_nopool_fm4_20261102_v1/stage4_rms/main_nfe4_100_v1/result.json`。

完整审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_stage4_nopool_fm4_20261102_v1/final_audit.json`；pool=true对照审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_four_variants_fm4_20261102_v1/final_audit.json`。

## 参数与运行记录

平均池化本身不含可训练参数；关闭后每视角token由4增至6，三视角由12增至18，注意力/KV长度相应增加。

| 配置 | 总参数 | 可训练参数 |
|---|---:|---:|
| stage4 | 360,604,940 | 359,922,062 |
| stage4_rms | 93,818,124 | 93,135,246 |

训练均无非有限loss/OOM/未恢复错误。stage4闭环无worker重启；stage4_rms闭环中固定seed100081、100108的专家场景检查曾失败，原runner自动按相同seed恢复worker共5次（1+4），随后完成。没有替换seed、跳过回合或因policy失败而重跑；原始错误日志保留，最终100唯一seed与100视频完整审计通过。

两条精确request均succeeded/exit0、两机均恢复holding；两组W&B实际服务端FINISHED/8771，CPU同步完成。参数及同seed恢复证据见同目录report_supplement.json。
