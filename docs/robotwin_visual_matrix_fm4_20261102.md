# RoboTwin visual backbone matrix — 2026-11-02

状态：运行中。以下为 **2026-11-03 14:07 Asia/Shanghai** 已核验的过程快照，并非32组最终报告。32组均为本次新训练；已完成23组训练和17组各100次闭环，其中16组已有聚合结果，另1组四分片齐全、等待同批统一聚合。

共同配置：GR00T式q01/q99分位数归一化并clip、三视角、50demo（49训练/1留出）、训练seed42、200epochs、global batch128、BF16训练和PyTorch eager闭环、标准FM、tau_token、1000桶、实际4次前向[0,250,500,750]、KV开启、CUDA Graph关闭。每组使用同任务100个固定唯一合法seed，保存100段非空视频。池化均在融合后；本矩阵明确同时比较池化与不池化，日常默认仍为不池化。

baseline：DiT/state/action/t为2048维，普通视觉MLP。新网络：冻结backbone及同一stage4额外复用两次、一个共享可训练桥接、融合器原五处LayerNorm、视觉RMSNorm＋残差MLP、DiT/state/action/t为1024维；非双MBConv。两者DiT FFN均等宽。B0/B1原生视觉宽度分别128/256；DINO Large为DINOv3 ConvNeXt-Large，仅比较baseline。

| Backbone | Task | Policy | Pooling | 训练状态 | 本次闭环结果 | 实际W&B API状态/步数 |
|---|---|---|---|---|---|---|
| b0 | click_alarmclock | baseline | 2×2 | 完成 6600/6600步 | 58/100 | FINISHED/6600 |
| b0 | click_alarmclock | baseline | 无 | 完成 6600/6600步 | 63/100 | FINISHED/6600 |
| b0 | click_alarmclock | stage4_rms | 2×2 | 完成 6600/6600步 | 79/100 | FINISHED/6600 |
| b0 | click_alarmclock | stage4_rms | 无 | 完成 6600/6600步 | 83/100 | FINISHED/6600 |
| b0 | adjust_bottle | baseline | 2×2 | 完成 11071/11071步 | 79/100 | FINISHED/11071 |
| b0 | adjust_bottle | baseline | 无 | 完成 11071/11071步 | 90/100 | FINISHED/11071 |
| b0 | adjust_bottle | stage4_rms | 2×2 | 完成 11071/11071步 | 77/100 | FINISHED/11071 |
| b0 | adjust_bottle | stage4_rms | 无 | 完成 11071/11071步 | 86/100 | FINISHED/11071 |
| b0 | blocks_ranking_rgb | baseline | 2×2 | 完成 35332/35332步 | 评测中 40/100回合 | FINISHED/35332 |
| b0 | blocks_ranking_rgb | baseline | 无 | 完成 35332/35332步 | 0/100 | FINISHED/35332 |
| b0 | blocks_ranking_rgb | stage4_rms | 2×2 | 完成 35332/35332步 | 待评测 | FINISHED/35332 |
| b0 | blocks_ranking_rgb | stage4_rms | 无 | 完成 35332/35332步 | 待评测 | FINISHED/35332 |
| dinol | blocks_ranking_rgb | baseline | 2×2 | 排队 | 待训练 | — |
| dinol | blocks_ranking_rgb | baseline | 无 | 排队 | 待训练 | — |
| dinol | beat_block_hammer | baseline | 2×2 | 排队 | 待训练 | — |
| dinol | beat_block_hammer | baseline | 无 | 排队 | 待训练 | — |
| b1 | click_alarmclock | baseline | 2×2 | 完成 6600/6600步 | 65/100 | FINISHED/6600 |
| b1 | click_alarmclock | baseline | 无 | 完成 6600/6600步 | 65/100 | FINISHED/6600 |
| b1 | click_alarmclock | stage4_rms | 2×2 | 完成 6600/6600步 | 73/100 | FINISHED/6600 |
| b1 | click_alarmclock | stage4_rms | 无 | 完成 6600/6600步 | 71/100 | FINISHED/6600 |
| b1 | adjust_bottle | baseline | 2×2 | 完成 11071/11071步 | 88/100 | FINISHED/11071 |
| b1 | adjust_bottle | baseline | 无 | 完成 11071/11071步 | 89/100 | FINISHED/11071 |
| b1 | adjust_bottle | stage4_rms | 2×2 | 完成 11071/11071步 | 47/100 | FINISHED/11071 |
| b1 | adjust_bottle | stage4_rms | 无 | 完成 11071/11071步 | 73/100 | FINISHED/11071 |
| b1 | blocks_ranking_rgb | baseline | 2×2 | 完成 35332/35332步 | 待评测 | FINISHED/35332 |
| b1 | blocks_ranking_rgb | baseline | 无 | 完成 35332/35332步 | 待评测 | FINISHED/35332 |
| b1 | blocks_ranking_rgb | stage4_rms | 2×2 | 完成 35332/35332步 | 待评测 | FINISHED/35332 |
| b1 | blocks_ranking_rgb | stage4_rms | 无 | 训练中 25953/35332步 | 待评测 | RUNNING/25525 |
| dinol | adjust_bottle | baseline | 2×2 | 排队 | 待训练 | — |
| dinol | adjust_bottle | baseline | 无 | 排队 | 待训练 | — |
| dinol | click_alarmclock | baseline | 2×2 | 排队 | 待训练 | — |
| dinol | click_alarmclock | baseline | 无 | 排队 | 待训练 | — |

完整覆盖100seed的已完成结果才填写成功数；进行中只写回合数。Blocks Ranking RGB的B0 baseline无池化目前实测0/100，固定seed、视频、运行设置及checkpoint身份核验通过，按原结果保留。训练步数依任务数据长度不同：click 6600、adjust 11071、blocks 35332、beat 8771，均200epochs。

machine1：B0三个任务12组，随后DINO blocks/beat4组。machine2：B1三个任务12组，随后DINO adjust/click4组。每组仅一次正式训练，先CPU真实数据预检、八卡两次forward/backward零optimizer检查和初始main/EMA严格重载，再正式训练、最终main/EMA严格重载及四分片闭环。两条4-GPU lane独立领取任务。当前DINO8组仍排队。

已完成训练的main/EMA strict、实际FM4、token/KV形状、2000→4000模块更新及冻结backbone匹配均已核验；封存1012份源码/配置及304份runner/config哈希一致。已完成的100次闭环逐条核验固定seed集合和100个非空视频。

运行记录：曾修复一次共享W&B反向SSH连接和两条CPU relay的传输故障，保留原run ID并补传，已完成训练的实际服务端状态和最终步数均核验完成。原runner出现三次同seed专家场景初始化恢复：B0 adjust新网络池化、B1 adjust新网络不池化的seed100011各一次，以及B0 blocks baseline不池化的seed100052一次；均在首次policy推理之前，未替换seed或因policy失败重跑。日志和产物全部保留。

计划、封存源码、配置、Job/request及各case路径：

- 战役证据根：`/data/vepfs/licheng.yang/100HzS1/runs/robotwin_visual_matrix_fm4_20261102_v1`。
- 训练路径：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/<task>/robotwin_visual_matrix_fm4_20261102_v1/<case>`。
- 闭环路径：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/<task>/robotwin_visual_matrix_fm4_20261102_v1/<case>/main_nfe4_100_v1`。
- 实验YAML：`configs/100hzs1_train/experiments/visual_matrix_20261102/`（base＋32组）。

用户于2026-11-03授权将当前代码和实验YAML提交、合并main并推送GitHub。运行中的两台机器继续使用原封存快照，提交和合并不改变其源码、配置或输出。全部32组结束后，将本文件补齐为最终审计结果并提交；本快照不宣称实验已经完成。
