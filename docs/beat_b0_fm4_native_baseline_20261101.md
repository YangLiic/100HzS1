# Beat B0 原始维度 baseline：标准 FM4

用户在降维组22/100后明确要求补跑原始、不降维的B0，使用machine1。
只新增YAML恢复hidden1024→2048、heads16→32、视觉Linear输出512→1024，
以及独立输出目录；网络/训练/数据/评测实现保持与已完成FM4降维组一致。

冻结预训练B0输出80token/视角、128维；原融合后ResNetAdapter两层ResBlock
及2×2池化，每视角4token；Linear128→1024，三视角展平12token，sceneMLP
1024→2048→GELU→2048。状态/动作MLP14→2048→GELU→2048→GELU→2048，
标准FM仅t时间嵌入2048，无r；DiT8层32heads64维，KV[B,32,12,64]。
不使用共享stage4额外调用、桥接或独立MBConv下采样。

共同设置：标准FM/tau_token，1000时间桶，4-step推理与评测；
beat_block_hammer/demo_clean，50demo，49训练/1验证，5614/118样本，seed42，
200epoch8771步，batch128，8L4，BF16，状态/动作minmax、三视角，
horizon24=1状态+23动作。离线BGR→RGB，闭环RGB不翻转，eager/KV-on/Graph-off。
只加载冻结B0预训练，策略重新训练，不恢复先前策略权重。

流程：真实数据CPU模型/配置/维度/4次DiT前向预检，八卡两步smoke与main/EMA
严格重载，正式训练，最终main/EMA严格重载，同机最终main100固定合法seed
闭环（4个25seed分片、两条独立4GPU热队列lane），保存100回合视频。
不追加EMA对比、NFE扫描或其他实验，不push。

Run: /data/vepfs/licheng.yang/100HzS1/runs/beat_b0_fm4_native_d2048_baseline_20261101_v1
Request: s1-beat-b0-fm4-native2048-20261101-v1
Machine1 Job: t-20260820125318-wd2mn
Checkpoint: /DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_fm4_native_d2048_baseline_20261101_v1
Eval: /DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_fm4_native_d2048_baseline_20261101_v1/main_nfe4_100_v1

复用beat-b0-fm4-ablation巡检，一并跟踪baseline和已授权的两组消融。
旧两组冻结源码与配置、产物不变；baseline从独立新快照运行。

## 最终结果（2026-11-01 21:37 Asia/Shanghai）

| B0 标准FM4实验 | 隐藏维度 | 每视角token | 成功数/100 |
|---|---:|---:|---:|
| 原始融合后两ResBlock+2x2池化 baseline | 2048 | 4 | **52/100** |
| 原压缩模块，Linear128->512、scene512->1024->1024 | 1024 | 4 | 22/100 |
| 融合前共享stage4额外两次，无末端池化 | 2048 | 6 | 48/100 |
| 融合后共享stage4替换两ResBlock，保留2x2池化 | 2048 | 4 | machine2训练中 |

Baseline训练完成8771步，最终loss0.00115，用时37:30。最终main和EMA均严格
重载通过，冻结B0在各自保存精度下匹配预训练；实际FM四次DiT前向为
[0,250,500,750]，无r分支。闭环只评测最终main，四分片成功数13+17+11+11，
恰好100个固定唯一seed、100个非空独立视频；eager/BF16/NFE4/KV-on/
Graph-off/RGB-no-flip及checkpoint身份、源码配置种子哈希全部核验。
完整审计 runs/beat_b0_fm4_native_d2048_baseline_20261101_v1/final_audit.json。

原870个网络、训练、数据和评测实现文件与降维组完全相同；本次YAML仅恢复
hidden1024->2048、heads16->32、视觉Linear512->1024。结果较降维组多30个
成功回合；与融合前共享stage4组相差4个回合。单次训练seed42和100回合结果
不说明小差异可稳定复现，也不与不同backbone或旧MeanFlow成绩混用。

W&B: https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/3dvbn2u7 ，
实际服务端FINISHED、step8771、末尾指标有限。精确request成功exit0，
2026-11-01T13:28:08UTC退出，machine1已恢复holding。

machine2最新融合后替换组独立继续；不重复启动已完成实验，不停止共享
W&B转发和巡检，待第四组最终审计完成后统一收尾。代码及报告未push。
