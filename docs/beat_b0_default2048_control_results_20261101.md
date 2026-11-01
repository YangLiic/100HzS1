# B0 恢复默认 2048 维：配置排除实验结果

2026-11-01，machine1，beat_block_hammer / demo_clean。

最终主模型闭环 **25/100（25%）**。同一套 B0、原融合后 2×2 压缩、MeanFlow 配方下，上一组 1024 维为 **18/100（18%）**，本次增加 **7 个百分点**。只进行了一次训练种子42的实验，不能据此证明所有低成功率都由维度造成，也不能排除所有实现或训练配方问题。

**训练目标仍为 MeanFlow，时间注入 tau_token，评测 NFE=1。这里“恢复默认”仅指恢复网络宽度，并未改回标准 FM。不同 backbone 或标准 FM 的历史成绩不能当作这次 B0 配方的直接复现基线。**

## 配置与网络

本次只新增一个 YAML，没有修改网络、训练、数据或评测实现。冻结源码中 870 个 src/scripts/simulation/main.py 文件与上一组 1024 维完全一致，982 个快照文件及全部封存 runner/config 哈希验证通过。

| 项目 | 上组 1024 维 | 本次 2048 维 |
|---|---|---|
| B0 编码输出 | 每视角 80×128 | 每视角 80×128 |
| 融合与压缩 | 原 128 维融合，ResNetAdapter 后池化到每视角2×2 | 相同 |
| 视觉投影 Linear | 128→512 | 128→1024 |
| 展平 | 三视角×4，12 token | 相同 |
| scene MLP | 512→1024→GELU→1024 | 1024→2048→GELU→2048 |
| DiT/状态/动作/t/r 隐藏维度 | 1024 | 2048 |
| DiT 层数 / heads / head_dim | 8 / 16 / 64 | 8 / 32 / 64 |
| 最终主模型成功数 | 18/100 | 25/100 |

三个实际模型配置差异仅为 hidden_size 1024→2048、num_heads 16→32、视觉投影 dim_out 512→1024。模型模块名称与类型拓扑一致，无新增 MBConv、repeat stage4 或入口桥接。B0 预训练且冻结。

实际 forward trace：输入 [1,3,3,240,320]，原图像预处理后 B0 输出每视角80个128维 token；融合 [1,3,80,128]；原压缩得到每视角4个 token；Linear [3,2,2,128]→[3,2,2,1024]；每视角特征 [1,3,4,1024]；展平 [1,12,1024]；scene MLP [1,12,2048]；KV 每层 K/V 均 [1,32,12,64]。状态/动作 MLP 为14→2048→GELU→2048→GELU→2048，t/r 同步2048。

共同训练设置：50 demo，49训练/1验证，5614/118样本，200 epoch，8771步，seed42，全局batch128，8×L4，BF16，AdamW lr1e-4，状态/动作 min-max，absolute joint14，三视角，horizon24（1状态+23动作）。MeanFlow time_embed_scale=1、equal_time_ratio=0、guidance_scale=1，tau_token。仅加载冻结B0预训练，不恢复完整策略旧权重。

模型总参数 365931006，可训练参数 365248398。

## 训练和最终权重验证

八卡真实数据两步 smoke、smoke main/EMA 严格重载、正式训练和最终 main/EMA 严格重载全部通过。训练8771/8771，最终进度日志 loss=0.0325，用时50:18，2.91步/秒。未发现 loss NaN/Inf、OOM、Traceback 或运行错误。

最终输出 action [1,23,14]，conditions [1,12,2048]；主模型及EMA的冻结B0都与预训练按各自保存精度匹配。主模型FP32与EMA BF16比较时使用保存精度转换。

- Main SHA256：cc11e89f1f1880c8a32b51cdbf7fecdb459e713fa01f3471e19856f3366886f4
- EMA SHA256：239d0818856232cea2e9722e5f980ee53b8de6721bbc0d6a1727caf1379751f6
- 训练配置 SHA256：3e51694935f7b805d3882555990202d3c253bf5792a9924896c8bf35698773bb
- 固定100 seed文件 SHA256：fca2035b8f623d52f980843f8f578710c520997db5c7f334f557ebec5a2ec8da

## 闭环与运行状态

最终 main 权重，MeanFlow NFE1，eager BF16，KV开启，CUDA Graph关闭，SAPIEN RGB不翻转。四个分片各25个固定合法seed，逐条核验100个唯一seed与原计划完全一致，100个不重复、非空视频均存在。未更换seed，未额外评测EMA或其他NFE。

| 分片 | 完成回合 | 成功 |
|---|---|---|
| shard0 | 25 | 6 |
| shard1 | 25 | 3 |
| shard2 | 25 | 6 |
| shard3 | 25 | 10 |
| 合计 | 100 | 25 |

唯一request `s1-beat-b0-d2048-control-20261101-v1` 于2026-11-01 19:01:11（Asia/Shanghai）成功退出，exit_code=0。machine1 Job `t-20260820125318-wd2mn` 已恢复holding。

W&B：[正式训练 run](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/pyzcd127)，服务端 finished，lastHistoryStep=8771，summary step=8771；11:03:32UTC再次独立查询确认，最近4条指标均有限。CPU转发曾在最后配置/summary文件上传时受网络主机白名单限制，扩展到实际观察到的W&B文件上传域名后自动完成，全部4个文件上传成功；没有改动训练或冻结源码。确认没有其他消费者后，仅清理了本实验的本地网络helper PID73941与SSH tunnel PID61914。

## 产物路径

- 冻结源码提交：`1f073b2fa4e75eae43bcef67885ca77daed11b13`。
- 新增配置：`/data/vepfs/licheng.yang/100HzS1/configs/100hzs1_train/backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_postfusion_2x2_meanflow_d2048_control_3v_8gpu_L4.yaml`。
- 运行计划、封存快照与日志：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_post2x2_d2048_control_20261101_v1`。
- 完整训练配置：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_post2x2_d2048_control_20261101_v1/configs/train.yaml`。
- 最终权重与 strict_reload.json：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_post2x2_d2048_control_20261101_v1`。
- 100回合汇总：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_post2x2_d2048_control_20261101_v1/main_nfe1_100_v1/result.json`。
- 完整审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_post2x2_d2048_control_20261101_v1/final_audit.json`。
- 审计脚本：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_post2x2_d2048_control_20261101_v1/final_audit_default2048_control.py`。
- W&B服务端证据：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_post2x2_d2048_control_20261101_v1/wandb_api_final.json`。

本次未 push，未 StopJob，未修改封存网络或运行源码，保留全部已有产物。
