# B0 四种视觉结构，待确认后重新训练

本次仅保留四个训练配置。四组都从 B0 预训练初始化、重新训练一次并评测，包含 baseline；不恢复旧策略权重。当前没有提交新训练或评测，先按用户要求确认下面四种结构。

## 四种计算路径

所有组的原始输入均为三视角 240×320 图像，沿用原尺寸对齐至 256×320 和图像归一化。原 EfficientViT-B0 预训练主干冻结，每视角输出 128×8×10，即 80 个 128 维 token。

| 配置文件 | 视觉路径（每视角） | 视觉条件输出 | 状态／动作／时间／DiT 宽度 |
|---|---|---|---:|
| `b0_baseline_fm4.yaml` | B0 80 → 三视角融合 → 原两个 ResBlock（80→20→6）→ 原平均池化 2×2 → Linear 128→1024 → 原 MLP 1024→2048→GELU→2048 | 12×2048 | 2048 |
| `b0_stage4_fm4.yaml` | B0 80 → 共享桥接和冻结 stage4 额外两次（80→20→6）→ 三视角融合 → 原平均池化 2×2 → Linear 128→1024 → 原 MLP | 12×2048 | 2048 |
| `b0_rms_fm4.yaml` | B0 80 → 三视角融合 → 原两个 ResBlock → 原平均池化 2×2 → Linear 128→1024 → RMSNorm 残差 MLP | 12×1024 | 1024 |
| `b0_stage4_rms_fm4.yaml` | B0 80 → 共享桥接和冻结 stage4 额外两次 → 三视角融合 → 原平均池化 2×2 → Linear 128→1024 → RMSNorm 残差 MLP | 12×1024 | 1024 |

**四组池化全部在融合之后**，每视角池化为 4 token，三视角展平为 12 token。stage4 两组没有原两个 ResBlock；保留原池化和 Linear。没有融合后调用 stage4，也没有新增独立 MBConv。

共享压缩具体为 `128×8×10 → P(128→64) → 同一 stage4 → 128×4×5 → 同一个 P → 同一 stage4 → 128×2×3`。P 是一个可训练、无 bias 的 1×1 Conv，共 8192 参数；三视角、两次调用共用。stage4 包含其原生下采样 MBConv 与 EfficientViT 模块，来自原 B0，不复制参数，保持冻结且 BatchNorm 为 eval。冻结权重仍允许梯度回传给 P。AdaptiveAvgPool2d 本身增加 **0 参数**。

RMSNorm 残差 MLP 位于视觉侧，令 Linear 输出为 u，计算：

`y = u + Linear(2048→1024)(GELU(Linear(1024→2048)(RMSNorm(u))))`

直连旁路保持原 u；RMSNorm 只作用于残差支路。只做一次普通端到端训练，没有教师、蒸馏、重构或第二阶段。

## DiT 与各类 token

DiT 的注意力、FFN、时间注入、KV 缓存实现均未改动。沿用原配置支持的 D：baseline/stage4 为 2048，rms/stage4_rms 为 1024。8 层、head_dim=64，分别 32/16 heads；每层 FFN 始终 `D→D→GELU→D`，没有独立 FFN 宽度开关。

状态和带噪动作沿用原三 Linear、两 GELU 的共享编码器 `14→D→D→D`；时间沿用原频率编码和 MLP→D。本批实验是 FM/tau_token，没有 MeanFlow 的 r 分支；保留项目原有 tau_AdaRMS 能力，未变动其实现。

每层视觉 K/V 形状分别为 `[B,32,12,64]` 或 `[B,16,12,64]`。这里 D 同时控制整个去噪主干的容量，不能仅凭状态／时间的原始维度较低保证 D=1024 不掉成功率，需以重新评测为准。

## 配置整理与实验范围

四个 YAML 位于 `configs/100hzs1_train/backbones/efficientvit_b0/tau_token/`。仅 `model.visual_variant` 选择 `baseline / stage4 / rms / stage4_rms`；其余三个入口继承 baseline、只改该值。宽度、head 数、视觉 MLP 和 ResBlock 数由此推导，矛盾的手动覆盖会报错。

已删除旧 token/降维 YAML 及归档 YAML，删除独立 MBConv、可选前融合 ResBlock 压缩、TokenProjection 和非线性输出投影分支。移除 repeat 次数、池化位置/形状、独立压缩层数、视觉残差中间宽度等实验开关。代码中旧名称仅用于拒绝过期配置，不能启用旧路径。

minmax 等归一化选项、其他 backbone 的常规配方以及项目原有 FM/MeanFlow、两种时间注入功能保持。此次四组共同使用 beat_block_hammer/demo_clean、50demo、seed42、200epoch（8771 步）、batch128、三视角、minmax、BF16、标准 FM/tau_token、1000 时间桶、4-step 推理、eager、KV-on、Graph-off；保留原数据颜色和缓存设置，闭环 RGB 不翻转。

确认结构后四组全部重新训练，各用最终 main 做相同固定 100 seed 的闭环评测并保存视频；main/EMA 权重均做严格重载，但不另加 EMA 对比或 NFE 扫描。machine1/2 安排在确认后制定，不重复启动历史实验。

## 清理与核验

旧 B0 实验的 96 份权重文件、400 个视频及对应训练缓存／RoboTwin 输出目录已删除，合计 46,570,212,398 字节。预训练 B0/B2/B3 文件哈希不变。保留 Git 历史、冻结源码和少量审计记录用于追溯，不能作为当前配置入口。

59 项 CPU 回归测试通过，覆盖四种视觉路径、共享权重/桥接梯度、配置与重载、残差 MLP、两种时间注入、KV 缓存和归一化。四种完整模型分别实测条件、状态、时间、K/V 维度及 FM 前向时间桶 `[0,250,500,750]`；均为有限值，没有 optimizer step。DiT 和归一化实现哈希与修改前完全一致。

远端核验目录：`runs/four_visual_variants_20261102_v1/`，包含 `architecture_verification.json`、`tests.log`、`old_artifact_deletion.json`。本说明对应新四配置，不复用旧实验成功率作为新结果。
