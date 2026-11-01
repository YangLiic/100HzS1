# Beat B0 降维消融1024：最终结果

machine1 的本次实验已完成：最终主模型在相同100固定seed上成功 **18/100（18%）**。旧 A 为0/100。两组均为B0、融合后原2×2压缩、MeanFlow/tau_token、200epochs、seed42、50demo、8×L4、BF16和相同评测设置；本次把投影宽256改512、隐藏宽512改1024，保留head维64（8heads改16heads）。本次结果高于旧 A，但只有一个训练seed，不能证明容量是旧低成功率的唯一原因。

## 结构核对

旧 A 实际checkpoint中的Linear128→256确实存在，scene MLP为256→512→GELU→512，state/action MLP为14→512→GELU→512→GELU→512，t/r和DiT宽512。最早compact256还改过压缩位置和scene identity，不能当成纯宽度结果。

本次严格保留原融合和ResNetAdapter：冻结预训练B0 →每视图80token/128维 →原多视图融合 →两ResBlock下采样 →自适应平均池化2×2 →Linear128→512 →[B,3,4,512]展平[B,12,512] →Linear512→1024 →GELU →Linear1024→1024。状态/带噪动作保留三层MLP，t/r输出1024；DiT8层16heads，每层K/V为[B,16,12,64]，物理动作14维。没有stage4重复、额外MBConv或MLP删除。

CPU审计逐项比对解析配置，只发现hidden_size、num_heads、视觉dim_out三处模型配置差异；实际模块名/类型拓扑与严格加载的旧 A 一致。真实前向和八卡两步短测均通过。本次源码冻结自bddd80143e707bb291aff42f9c7f8b58a6d56801，共核验981个源码文件。

## 训练与评测核验

- 正式8771/8771步，最终loss 0.0275，训练日志无NaN/Inf/OOM/Traceback；报告时长27分09秒。
- main与EMA最终严格重载成功，输出有限动作[1,23,14]；冻结B0按各checkpoint保存精度与预训练精确一致，配置/源码/权重哈希均匹配。
- 最终main，MeanFlow NFE1、eager BF16、KV开启、Graph关闭、SAPIEN RGB不翻转。
- 四分片成功数3/25、5/25、5/25、5/25；100个固定唯一seed覆盖精确相等，100个独立非空视频逐一核验。expert初始化重试保留相同seed，没有替换seed。
- request成功exit0，结束于2026-11-01 17:35:00（Asia/Shanghai）；machine1恢复holding。
- W&B服务端实际8771步、state=finished，CPU实时转发完整结束。[训练曲线](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/vvbfx544)。

## 路径与复现

- 权重：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_post2x2_d1024_20261101_v1`；main为`model.safetensors`，EMA为`ema/model.safetensors`。
- 闭环汇总和逐回合视频目录：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_post2x2_d1024_20261101_v1/main_nfe1_100_v1/result.json`。
- 冻结源码、配置、队列与完整审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_post2x2_d1024_20261101_v1`；解析配置`configs/train.yaml`，最终审计`final_audit.json`。
- source commit：`bddd80143e707bb291aff42f9c7f8b58a6d56801`。
- main SHA256：`516bdb833134dfb344ef230de52cd9dd1e5a39b2363e0bf361d10b7a9923c831`。
- EMA SHA256：`840313adc00cce29c8e4d03245bf1b68be0341c54ae29671ad247ae8bed3fc91`。
- seed SHA256：`fca2035b8f623d52f980843f8f578710c520997db5c7f334f557ebec5a2ec8da`。
- 训练配置 SHA256：`67931328ef9f295d01560ad91dc4d36e3328239ec382b4d22f4138fbaaa16659`。

未追加实验或评测，未push。machine2的独立MBConv实验仍由它自己的巡检负责。
