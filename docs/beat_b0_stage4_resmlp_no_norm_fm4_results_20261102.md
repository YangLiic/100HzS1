# B0共享stage4＋残差MLP：去掉视觉RMSNorm消融结果

本次新训练已完成，主模型闭环 **38/100**；此前保留RMSNorm的同配对照为 **57/100**，减少 **19个百分点**。这次结果支持在当前结构中保留视觉RMSNorm，不能将它视为已证实的冗余层。两组各只有一个训练seed，结论限于本次对照。

| 配置 | 成功数/100 | DiT宽度 | 视觉token | 总参数 | 可训练参数 | W&B |
|---|---:|---:|---:|---:|---:|---|
| 保留RMSNorm＋残差MLP（已有对照） | 57 | 1024 | 18 | 93,818,124 | 93,135,246 | [FINISHED 8771](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/tso2flyl) |
| 仅残差MLP，去掉视觉RMSNorm（本次） | 38 | 1024 | 18 | 93,817,100 | 93,134,222 | [FINISHED 8771](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/4rzy8t5b) |

## 控制变量与真实网络

本次只移除视觉残差支路中的RMSNorm，保留其余架构及全部宽度；解析模型配置唯一差异为scene_adaptor从residual_mlp2x_gelu改为residual_mlp2x_gelu_no_norm，总共减少1024个缩放参数。同seed初始化时，其余视觉MLP参数逐值一致。对照57/100没有重新训练。

原冻结B0 → 同一个冻结stage4额外两次（共享可训练128→64桥接）→ 每视角6token/128维 → 三视角融合及原LayerNorm → 不池化 → Linear128→1024 → 展平为18token → x+W₂GELU(W₁x)。视觉MLP仍1024→2048→1024。DiT/state/action/t均1024，8层16heads，DiT内部FFN仍固定等宽。FM无r分支。

第一项展平位置消融按用户要求取消：实现位置保持不变，没有为此训练策略。此前只完成局部CPU等价性检查，未把该检查计作正式策略训练。

共同使用beat_block_hammer/demo_clean、50demo（49训练/1留出）、5614/118样本、seed42、200epoch/8771 optimizer steps、batch128、三视角、minmax、BF16，视频/RAM缓存及离线BGR→RGB设置沿用对照。标准FM/tau_token/1000桶；闭环main、NFE4、eager/BF16/KV开启/Graph关闭/RGB不翻转。

## 同一100固定seed配对

| 结果 | 回合数 |
|---|---:|
| 两组都成功 | 29 |
| 仅保留RMSNorm成功 | 28 |
| 仅去掉RMSNorm成功 | 9 |
| 两组都失败 | 34 |

37个不一致回合中28比9；双侧精确McNemar检验p=0.00256321。检验针对这两个已训练策略在固定seed集合上的配对结果，不替代多训练seed重复，也不证明所有任务上都需要该归一化层。

## 完成核验

14项CPU测试、真实数据预检及八卡两次forward/backward零optimizer启动检查通过。正式训练唯一一次8771步，最终loss0.00334，无非有限loss/OOM/训练异常。初始与最终main/EMA均严格重载通过；实测FM前向时间[0,250,500,750]、条件[1,18,1024]、KV[1,16,18,64]，明确visual_rms_norm_enabled=false，无scene_adaptor.norm参数。

2000→4000检查点的桥接、视觉MLP、Linear、融合器及DiT FFN确有更新；两检查点main/EMA中原B0全部186张量均按保存精度匹配预训练。封存972份源码/配置、训练配置、预训练和最终权重SHA核验通过。冻结提交 `3c8ce1913861824693d137dfddbfe00718c586cd`。

四分片均完整25回合，覆盖100唯一固定seed与100非空唯一视频。本次闭环无worker重启或错误记录，未替换seed、跳过回合或因policy失败重跑。W&B真实服务端FINISHED8771，CPU转发完成。唯一request `s1-beat-b0-resmlp-no-norm-m1-20261102-v1` 已succeeded/exit0，machine1恢复holding。

## 产物路径

- 正式训练/主权重/EMA：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_stage4_resmlp_no_norm_fm4_20261102_v1/stage4_rms`。
- 本次闭环结果及各分片视频：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_stage4_resmlp_no_norm_fm4_20261102_v1/stage4_rms/main_nfe4_100_v1/result.json`。
- 完整审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_stage4_resmlp_no_norm_fm4_20261102_v1/final_audit.json`。
- 控制变量证据：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_stage4_resmlp_no_norm_fm4_20261102_v1/controlled_config_comparison.json`。
- 同seed配对与运行记录：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_stage4_resmlp_no_norm_fm4_20261102_v1/paired_comparison.json`、`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_stage4_resmlp_no_norm_fm4_20261102_v1/report_supplement.json`。
- 已有57/100对照审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_stage4_nopool_fm4_20261102_v1/final_audit.json`。
