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
