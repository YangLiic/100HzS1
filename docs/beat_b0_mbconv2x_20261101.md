# Beat B0：独立可训练 MBConv 下采样

本次只运行一组实验：原 EfficientViT-B0 从本地预训练 safetensors 严格加载并冻结，新增两层独立可训练 MBConv，下采样后再视觉融合。沿用 beat_block_hammer、50 demos、200 epochs、MeanFlow、tau_token、seed 42 和 machine2 八张 L4。

视觉路径：240×320 图像按原预处理变成 256×320；B0 输出每视图 128×8×10。新增 MBConv ×2 输出依次为 128×4×5、128×2×3。三视图以每视图 6 token 进入融合，随后 Linear 128→1024，scene MLP 1024→2048→2048（GELU）。不使用 ResNetAdapter 或自适应池化。

新增 MBConv 分别复制 stage4 两个 local MBConv 的预训练参数作为初始化：1×1 升维 128→512，3×3 depthwise（改为 stride=2），1×1 降维 512→128。它们不共享参数，不复制 LiteMLA，不复制原 stride=1 的残差支路，也不增加 128→64 桥接。原 B0 的参数和 BatchNorm 统计冻结；新层参数及 BatchNorm 统计参与训练。复制在显式加载预训练权重后执行，完整模型重载时直接严格加载已训练的新层。

DiT 保持 2048 维、8 层、32 heads；state/action MLP 保持 14→2048→2048→2048；t/r embedding 保持 2048 维。三视图产生 18 个条件 token，每层 K/V 为 [B,32,18,64]。

正式训练全局 batch 128、BF16、AdamW 1e-4、200 epochs（5614 训练样本，共 8771 步），不从之前实验的完整权重恢复。先进行八卡两步真实数据 smoke，并严格重载 main/EMA；正式训练结束再次严格重载 main/EMA，再用最终 main 在 machine2 跑相同的 100 个固定 expert-verified seed。闭环配置 MeanFlow NFE1、eager BF16、KV 开、CUDA Graph 关，SAPIEN RGB 不翻转，每回合保存视频。

配置入口：`configs/100hzs1_train/backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_mbconv2x_2x3_meanflow_d2048_3v_8gpu_L4.yaml`。

验证：6 项新增 CPU 测试覆盖配置、预训练复制与参数独立性、两次真实 MeanFlow 更新的梯度、尺寸与归一化模式、融合 token 数、严格重载；22 项已有视觉和 compact policy 回归测试通过。提交后以 git archive 冻结源码；正式训练使用独立输出路径和单一 machine2 request。结果另行记录，不覆盖旧实验。

补充：八卡短测时核查发现 EMA 原有实现只更新参数，不同步 BatchNorm 的运行均值、方差、计数。新增可训练 MBConv 需要这些统计，因此同步 BN buffers。原冻结 B0 的 main 权重必须与预训练逐位相等，EMA 按原有 BF16 保存精度与预训练转换后的值精确比较。首次短测产物保留，新 request/output 重新验证。
