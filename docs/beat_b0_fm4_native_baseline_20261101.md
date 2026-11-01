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
