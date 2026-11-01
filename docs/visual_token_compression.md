# EfficientViT 融合前视觉 token 压缩

新配置在每个视图独立完成空间压缩后，再进行多视角融合。使用现有
ResNetAdapter 的两个 stride=2 ResBlock，直接保留自然输出的 2×3 网格。
新路径不调用自适应池化，也不继续压缩为 2×2。

## 维度与计算顺序

以 B0、三视图、单帧图像观测为例，B 为观测 batch：

| 步骤 | 维度 |
| --- | --- |
| RoboTwin 原始图像 | `[B,3,3,240,320]` |
| 既有尺寸对齐和 ImageNet 归一化 | `[B,3,3,256,320]` |
| EfficientViT-B0 主干 | `[B,3,80,128]`，空间网格 8×10 |
| 第一个 ResBlock | 每视图 `[128,4,5]` |
| 第二个 ResBlock | 每视图 `[128,2,3]` |
| MultiViewFuser | `[B,3,6,128]`，每视图注意力长度 6，全局注意力长度 18 |
| 原 adapter 的 Linear，移到融合后 | `[B,3,6,1024]` |
| VisualEncoder 展平视图 | `[B,18,1024]` |
| 现有 scene_adaptor | `[B,18,2048]`，使用当前正式配方的隐藏维度 |

B2/B3 在融合与压缩期间分别保持 384/512 原生通道，末尾仍投影至 1024。
多帧图像观测按帧独立编码，DiT 条件长度自动变为 `n_image_obs_steps×V×6`；
位置嵌入长度由实际配置重新生成，没有硬编码成 18。

`shape_out: [2,3]` 在这条路径中用于校验自然下采样结果，不执行重采样。
改变图像大小或下采样层数导致网格不匹配时，会给出明确错误。

## 配置与使用

新开关位于 `model.obs_encoder.emvis_config`：

```yaml
compress_before_fusion: true
efficientvit_fusion_mode: native
model_adapter_config:
  target: ResNetAdapter
  conv_num: 2
  shape_out: [2, 3]
  dim_out: 1024
```

提供三个 observation 配置：

- `observation_config/efficientvit_b0_prefusion_2x3_3v.yaml`
- `observation_config/efficientvit_b2_prefusion_2x3_3v.yaml`
- `observation_config/efficientvit_b3_prefusion_2x3_3v.yaml`

B0/B2 分别提供独立的 beat_block_hammer MeanFlow 训练入口：

```bash
# 仅打印启动命令，不提交训练。
NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 scripts/train/train.sh \
  configs/100hzs1_train/backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_prefusion_2x3_meanflow_3v_8gpu_L4.yaml \
  --dry-run
```

B2 入口为同样目录结构下的
`backbones/efficientvit_b2/tau_token/100hzs1_beat_block_hammer_efficientvit_b2_prefusion_2x3_meanflow_3v_8gpu_L4.yaml`。
配方继承现有 B2 MeanFlow 的数据、BGR→RGB、观测、horizon 和训练设置，
仅调整视觉配置和独立输出路径。再次运行时另设 `output_dir`，避免覆盖产物。

未设置开关时仍是原来的“80 token 融合 → ResNetAdapter → 2×2 token”流程；
已有 observation 配置和训练入口没有改动。新模式仅支持 EfficientViT native
融合，并要求压缩后的通道数保持原生宽度。

## 参数与 checkpoint

本次复用原 ResNetAdapter 的代码结构，将其已有 Linear/Dropout 移到融合后，
没有增加一套投影，也没有额外重复 EfficientViT Stage 4。三视图共用同一个
视觉主干和同一个压缩器。视觉模块参数总量与对应的原结构相同。

Stage 4 无法直接与新增下采样共享整段权重：B0 的原 Stage 4 输入为 64 通道，
输出为 128 通道。其内部 128→128 的 stride=1 EfficientViT block 可以再次调用，
但不会减少 token；它不能替代这里的 stride=2 压缩层。

主干仍从本地预训练文件初始化并默认冻结，新的压缩/融合/投影和策略层需要训练。
新 checkpoint 保存 `compress_before_fusion`，可离线重建并 strict reload。
原配置默认值、原权重键和原输出保持可加载。将旧 2×2 策略改为本模式属于新架构：
输出投影权重键和 DiT 的视觉位置嵌入形状会改变，不能直接视为旧实验的无缝续训。
新训练入口明确设置 `resume_from_checkpoint: null`。

## 验证

2026-10-31，开发机 CPU / PyTorch 2.7.1 / timm 1.0.15：

- 新增 8 项测试：B0/B2/B3 实际融合输入为每视图 6 个 token、禁止调用池化、
  单视图及外部 token 入口、压缩/融合/投影梯度、冻结 BN、配置组合、参数数量、
  完整 MeanFlow loss/采样/KV 缓存和严格重载。
- 原有视觉编码器 10 项测试通过。
- 推理后端、KV 缓存、checkpoint 元数据 19 项回归测试通过。
- 小型完整策略的 encoder 导出通过 ONNX checker 和 ONNX Runtime CPU 对照；
  条件形状 `[1,18,64]`，最大绝对误差约 4e-7，状态归一化误差为 0。
  此处使用小型 DiT 验证接口，正式训练配方仍为 2048 隐藏维度。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=tests .venv/bin/python -m unittest \
  test_visual_compression test_visual_backbones test_inference_backends \
  test_kv_cache test_100hzs1_checkpoint_metadata -v
```

本次未提交正式训练、未测量 GPU/TensorRT 时延或 RoboTwin 成功率。
融合序列变短是结构事实，端到端速度与任务精度需要训练后另做实测。

## 256 维对齐与非线性视觉投影（2026-11-01）

新增 compact 训练入口，将视觉条件、状态/动作 token、时间嵌入、DiT 隐藏层
统一为 256 维。前述 1024→2048 是原入口的结构，仍保留用于对照；compact
入口不会经过 1024/2048 维瓶颈。

| 部分 | 原入口 | compact 入口 |
| --- | --- | --- |
| B0 编码/压缩/融合 | 128 维，每视图 6 token | 保持相同 |
| B0 融合后视觉投影 | Linear 128→1024 | Linear 128→256 → GELU → Linear 256→256 |
| scene_adaptor | Linear 1024→2048 → GELU → Linear 2048→2048 | Identity，视觉特征已是 256 维 |
| state_adaptor | Linear 14→2048 → GELU → Linear 2048→2048 → GELU → Linear 2048→2048 | Linear 14→256 → GELU → Linear 256→256 → GELU → Linear 256→256 |
| DiT | 8 层，2048 维，32 heads | 8 层，256 维，4 heads |
| 每层 K/V 联合投影 | Linear 2048→4096 | Linear 256→512，K、V 各 256 维 |
| 每层每个 K / V | `[B,32,18,64]` | `[B,4,18,64]` |
| 真实关节状态/输出动作 | 14 维 | 保持 14 维 |

state_adaptor 没有删除 MLP：仍有三层 Linear 和两处 GELU，激活在最后一层前。
采样时先将观测状态填入 trajectory 前缀，再将剩余位置的噪声动作与状态前缀
一起逐 token 通过同一个 state_adaptor。它们共用权重，随后加入时间/位置条件
进入 DiT。缩小的是内部特征表示，而非真实关节数。

视觉 MLP 位于 `scene_encoder.output_projection`，采用 GELU(tanh)，第二层后
没有额外激活。原 scene_adaptor 在 compact 配方中为 Identity，避免再增加
一套 256→256 的 MLP。B2 同样支持此配方，首层投影为其原生 384→256。

共享配置为 `policy_config/compact_width.yaml`：

```yaml
model:
  hidden_size: 256
  num_heads: ${eval:'${model.hidden_size}//64'}
  scene_adaptor: identity
  state_adaptor: mlp3x_gelu
  obs_encoder:
    emvis_config:
      output_projection_type: mlp2x_gelu
      model_adapter_config:
        dim_out: ${model.hidden_size}
```

compact 的 B0/B2 配方文件名在 `meanflow` 后增加 `_compact`，例如：

```bash
NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 scripts/train/train.sh \
  configs/100hzs1_train/backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_prefusion_2x3_meanflow_compact_3v_8gpu_L4.yaml \
  --dry-run
```

可通过 `model.hidden_size=128` 或 `model.hidden_size=512` 选择对照宽度，
视觉输出宽度、DiT heads 和输出目录中的 d128/d256/d512 标识同步变化，
每个 DiT head 保持 64 维。配置默认仍为用户选择的 256 维。

`output_projection_type` 会写入新 checkpoint；缺省仍为原先的 linear，
保持旧结构权重键。Identity 条件适配器会显式检查输入输出宽度相等。
compact 结构需要新训练，不能 strict 加载 2048 维策略或恢复其优化器。
新入口保持 `resume_from_checkpoint: null`，使用独立输出目录。

验证覆盖 128/256/512 配置、B0/B2 前向、状态/视觉各层梯度、FM/MeanFlow/
tau_AdaRMS 的两次 CPU 优化器更新、冻结主干、KV 等价性、完整策略 strict
reload，以及原有视觉/推理/时间注入回归。共 50 项 CPU 测试通过：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTHONPATH=tests .venv/bin/python -m unittest \
  test_compact_policy test_visual_compression test_visual_backbones \
  test_inference_backends test_kv_cache test_100hzs1_checkpoint_metadata \
  test_time_injection -v
```

完整 8 层、256 维模型的 encoder（含 KV 构建）及 DiT step 均通过 ONNX
checker 和 ONNX Runtime CPU 对照：导出缓存形状为 `[1,8,2,4,18,64]`，
KV 最大绝对误差约 2.27e-6，DiT step 最大绝对误差约 4.77e-7，
状态归一化误差为 0。此验证使用随机初始化模型检查实现和导出一致性。

这些检查不代表正式训练后的任务精度，也未测量 GPU/TensorRT 时延。

## 两种 tau 注入的 compact 配置

`tau_token` 与 `tau_AdaRMS` 都使用同一个 `model.hidden_size=256`。
通用模块本身按 hidden_size 构造，时间路径没有固定的 2048 维层。
MeanFlow 的 t/r 两个时间嵌入器均为 sin/cos 频率特征 256 → Linear 256→256
→ SiLU → Linear 256→256，两个向量相加后仍为 256 维。

| 路径 | tau_token | tau_AdaRMS |
| --- | --- | --- |
| 时间条件宽度 | 256 | 256 |
| 状态/动作序列 | 加一个 256 维时间 token，`[B,25,256]` | 不加时间 token，`[B,24,256]` |
| 动作与时间融合 | 不使用该模块 | 仅未来 23 个动作位置拼接为 512 维，再 Linear 512→256 → GELU → Linear 256→256 |
| 每个残差子层的调制 | 不使用该模块 | SiLU → Linear 256→768，拆成各 256 维的 scale/shift/gate |
| 输出端调制 | 不使用该模块 | SiLU → Linear 256→512，拆成各 256 维的 scale/shift |
| 每层视觉 K/V | `[B,4,18,64]` | `[B,4,18,64]` |

768 和 512 是多个调制参数拼接的总宽度，不改变 token 的 256 维隐藏表示。
MeanFlow 连续时间仍使用 `time_embed_scale: 1.0`。

B0/B2 均提供两种可直接选择的训练入口：

- `backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_prefusion_2x3_meanflow_compact_3v_8gpu_L4.yaml`
- `backbones/efficientvit_b0/tau_AdaRMS/100hzs1_beat_block_hammer_efficientvit_b0_prefusion_2x3_meanflow_adarms_compact_3v_8gpu_L4.yaml`
- B2 对应入口将路径和文件名中的 `b0` 替换为 `b2`。

AdaRMS 配方继承同一 compact 宽度、视觉结构和数据配方，只切换时间注入，
使用独立的 `_adarms_d256_` 输出目录。选择 128/512 维对照时，所有时间注入层
也跟随该隐藏维度变化；原始 sin/cos 频率特征保持 256 维，再投影到目标宽度。

补充的 compact/time-injection 两套测试共 14 项通过，覆盖实际时间 token 长度、
动作与时间融合输入、所有残差层的 scale/shift/gate、末层调制、时间路径梯度、
两种注入方式的完整策略 strict reload 和 cached/uncached 采样一致性。

256 维 AdaRMS 的完整 DiT step 还通过 ONNX checker 与 ONNX Runtime CPU
对照，最大绝对误差约 3.88e-7。该检查将 scale/shift/gate 与末层调制权重
设为非零，避免只验证零初始化的简化路径；未启动正式训练。
