# Beat B0：标准 FM、4-step 的独立消融

用户要求重新训练，只做两组；未明确指定 MeanFlow 时默认标准 FM、4-step。
这两组均为标准 `model.flow_type: fm`，1000桶训练时间，tau_token，4-step推理。
不使用先前 MeanFlow 策略权重。

| 实验 | 机器 | 视觉压缩 | 视觉与策略维度 |
|---|---|---|---|
| width1024_post2x2 | machine1 / t-20260820125318-wd2mn | 原128维融合后 ResNetAdapter + 每视角2×2；三视角12token | Linear128→512，sceneMLP512→1024→GELU→1024；state/action/t/DiT1024，8层16heads |
| stage4_d2048 | machine2 / t-20260819215926-zt9qr | 原B0冻结，共享同一个stage4额外调用两次，共享可训练1×1桥接128→64；80→20→6token/视角后融合；三视角18token | Linear128→1024，sceneMLP1024→2048→GELU→2048；state/action/t/DiT2048，8层32heads |

stage4压缩组不启用独立 MBConv 下采样，不追加2×2池化。B0预训练主干包括stage4被冻结；训练桥接、融合、投影及策略层。降维组保留原视觉融合、压缩与三层状态/动作MLP，只改变特征宽度。

共同设置：beat_block_hammer/demo_clean，50demo（49训练/1验证），seed42，200epoch，8771 optimizer steps，batch128，8L4/BF16，状态与动作min-max、absolute joint14，三视角，horizon24=1状态+23动作；离线BGR→RGB，闭环SAPIEN RGB不翻转。eager、KV开启、CUDA Graph关闭。仅加载冻结B0预训练，每组从新策略头开始。

标准 FM 只有 t 时间嵌入，r_embedder=None；r 仅用于 MeanFlow。t 与状态/动作/DiT 维度一致。

两台机器独立运行真实数据八卡两步smoke及main/EMA严格重载，再正式训练、最终main/EMA严格重载、同机最终main的100固定seed闭环。严格重载还记录真实DiT前向次数4及时间桶[0,250,500,750]，避免把MeanFlow或1-step混入。

复用已验证的100个expert合法seed，每组四个互不重叠的25seed分片，由两条独立4GPU lane热队列调度。逐回合保留视频，不替换seed，不追加其他baseline、EMA对比、步数扫描或双MBConv实验。

运行与冻结快照：
`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_fm4_ablation_20261101_v1`。

请求分别为：
`s1-beat-b0-fm4-width1024-20261101-v1` 与
`s1-beat-b0-fm4-stage4-20261101-v1`。

权重根：
`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_fm4_ablation_20261101_v1/<case>`。

闭环根：
`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_fm4_ablation_20261101_v1/<case>/main_nfe4_100_v1`。

用户授权清理了6轮旧B0降维/压缩实验：23个运行/权重/临时/评测目录、11个旧YAML、7份旧实验文档，逻辑大小58.386GiB；清单和删除回执在本次运行根的cleanup_manifest.json和cleanup_result.json。本地27份旧B0 YAML/结果副本也已清理。预训练、数据集、原生B0 profile、Git历史和其他实验保留。模型/训练/数据/评测实现870个文件哈希不变；只更新实验YAML、关联文档及不再依赖删除配方的测试fixture。

CPU回归26项通过。正式GPU结果需以本次独立产物及最终审计为准。W&B转发正式训练日志并核验服务端step；不把本地离线日志当上传成功。不push。


## 已提交与初始验证（2026-11-01 19:24 Asia/Shanghai）

冻结运行源码提交cc6dc4582debec7e61f15b91bd0e3845a09cf129。
两台机器均于11:21:35UTC领取唯一request，八卡两步真实数据smoke均通过。
main和EMA严格重载均确认标准FM、4次真实DiT前向、时间桶[0,250,500,750]、
有限action[1,23,14]、冻结B0预训练匹配，以及各自预期维度。

正式训练已分别于11:22:19UTC（machine1）和11:22:31UTC（machine2）启动。
初始进度919/8771与237/8771，未见非有限loss、OOM或运行错误。
完整快照973文件、sealed runner/config/预训练哈希均校验通过。

正式训练W&B（已独立查询实际服务端进度，非仅本地queued计数）：
- [降维组pt5tbit0](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/pt5tbit0)，初始服务端step651，running。
- [共享stage4组bnjepzv2](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/bnjepzv2)，初始服务端step84，running。

自动巡检beat-b0-fm4-ablation已建立，覆盖训练、最终main/EMA严格重载、
FM4闭环与W&B最终核验。健康进度保持安静，完成或失败时通知。旧巡检已删除。
