# B0 共享Stage4 + 普通MLP1024，无池化：FM4结果

本次仅训练新组一次；50/100与57/100是此前已完成的相同固定100seed对照，没有重启。

| 配置 | 视觉通道路径 | DiT维度 | token/视角（合计） | 成功数/100 |
|---|---|---:|---:|---:|
| 共享Stage4，无池化（旧对照） | 128→1024→2048→GELU→2048 | 2048 | 6（18） | 50 |
| 共享Stage4+RMSNorm残差MLP，无池化（旧对照） | Linear128→1024；RMSNorm+残差1024→2048→1024 | 1024 | 6（18） | 57 |
| 共享Stage4+普通MLP，无池化（本次） | Linear128→512；普通MLP512→1024→GELU→1024 | 1024 | 6（18） | 43 |

原直接降维22/100使用原两个ResBlock+融合后2×2池化、12token；不同于本次共享Stage4+无池化18token。

共同设置：beat_block_hammer/demo_clean，50demo（49训练/1留出），三视角singleframe，seed42，200epoch/8771optimizersteps，batch128，minmax，BF16，标准FM/tau_token/1000桶，推理4steps[0,250,500,750]，无MeanFlow r分支。闭环最终main，eager/BF16/KV-on/Graph-off/RGB不翻转；四分片各25个相同固定合法seed，100唯一seed和100非空视频。

冻结预训练B0；同一冻结Stage4额外复用两次，80→20→6tokens/view；同一个共享可训练1×1Conv128→64桥接8192参数；融合模块保持。融合后不池化，Linear128→512，展平为18×512，再普通Sequential[Linear512→1024,GELU,Linear1024→1024]。本次普通MLP没有RMSNorm与残差连接；这不是恢复历史的残差MLP去RMSNorm实验。state/action/t/DiT1024、8层16heads，DiT FFN始终1024→1024→GELU→1024。

总参数91,128,588，可训练90,445,710；冻结commit `6d9a6b50398e243470ec1e3fbea4348aa3608130`，972个源码/配置哈希核验。CPU预检、八卡真实数据两次forward/backward（零optimizer）、初始和最终main/EMA严格重载、真实四次FM采样、2000→4000参数真实更新与186个B0冻结张量核验均通过。正式训练一次，无教师/蒸馏/分阶段。

最终训练loss 0.00677；W&B [finished / 8771](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/snf2l186)，服务端实际步数已验证。

## 相同seed配对

| 旧对照 | 两者成功 | 仅本次成功 | 仅旧对照成功 | 两者失败 |
|---|---:|---:|---:|---:|
| stage4 | 29 | 14 | 21 | 36 |
| stage4_rms | 29 | 14 | 28 | 29 |

闭环同seed场景恢复事件数：5。原runner只按相同seed恢复，未替换seed、跳过回合或因policy失败重跑挑结果；全部原始日志保留。

每组仅训练seed42一次，100回合计数不代表多训练seed稳定均值。

## 产物路径

- 训练：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_stage4_plain_mlp1024_nopool_fm4_20261102_v1/stage4_mlp1024`
- 评测：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_stage4_plain_mlp1024_nopool_fm4_20261102_v1/stage4_mlp1024/main_nfe4_100_v1/result.json`
- 审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_stage4_plain_mlp1024_nopool_fm4_20261102_v1/final_audit.json`
- 配对/参数/场景恢复：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_stage4_plain_mlp1024_nopool_fm4_20261102_v1/report_supplement.json`
