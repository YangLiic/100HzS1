# VFA EfficientViT 接入与首轮实测

观测日期：2026-10-28。编码器适配和时延对比已完成。首轮训练已结束；当前重训使用单帧观测、horizon24。开环误差可核对输入与动作对齐，不能替代闭环任务成功率。

## 选择与结论

本轮选择 **MIT EfficientViT-B2-r256 ImageNet-1K 预训练权重**训练，另提供 B3-r256 配置用于切换。选 B2 的依据是本次 L4 实测的训练批次视觉前向速度和参数量；ImageNet 准确率不能代替机器人任务成功率。

B2 的完整视觉编码在 batch=16 时从 Base 的170.866ms降到81.709ms，即2.09倍速度、耗时降低52.2%；batch=1则为12.784ms，对比Base12.631ms，**当前 eager PyTorch 实现没有单次推理时延收益**。该结果不等于整个训练或整次动作预测加速2.09倍。

权重来自论文作者训练、timm 移植的分类模型，保留空间特征主干，去掉分类头。与 DINOv3 的自监督预训练不同，表示质量差异需闭环测量。来源：[MIT 分类模型说明](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_cls/README.md)、[B2 模型卡](https://huggingface.co/timm/efficientvit_b2.r256_in1k)、[B3 模型卡](https://huggingface.co/timm/efficientvit_b3.r256_in1k)。官方 Jetson/A100 TensorRT FP16 表格和这里的 L4 PyTorch BF16 条件不同。

## 维度与分辨率

| 项目 | DINOv3 ConvNeXt-Base | EfficientViT-B2 | EfficientViT-B3 |
|---|---|---|---|
| 读取图像，CHW | 3×240×320 | 相同 | 相同 |
| 实际输入主干 | 3×256×320 | 相同 | 相同 |
| 视角 / 图像历史 | 左、右、头三视角 / 1帧 | 相同 | 相同 |
| 最终空间下采样 | 32 | 32 | 32 |
| 空间网格 / 每视角tokens | 8×10 / 80 | 相同 | 相同 |
| 原生通道 | 1024 | 384 | 512 |
| 进入融合器前通道对齐 | Identity | 可训练Linear(384,1024) | 可训练Linear(512,1024) |
| 对齐层参数 | 0 | 394240 | 525312 |
| 主干参数（无分类头） | 87566464 | 14977008 | 38997440 |
| 完整视觉模块参数 | 145279104 | 73083888 | 97235392 |
| MultiViewFuser | 宽1024、16heads | 相同 | 相同 |
| ResNetAdapter | conv_num=2，每视角2×2tokens，通道1024 | 相同 | 相同 |
| 交给策略的视觉特征 | [B,12,1024] | 相同 | 相同 |

240→256来自原有VFA `preprocess_images(...,patch_size=32)` 的对齐规则；本次沿用该规则，没有改成224方图或使用分类模型默认裁剪。三种编码器逐像素核对了共同预处理结果；归一化均值[0.485,0.456,0.406]、标准差[0.229,0.224,0.225]，不二次除255。离线数据先按原有 `bgr_to_rgb=true` 纠正，模型输入RGB。

主干默认冻结。EfficientViT的BatchNorm也固定在eval模式，防止外层policy.train()改变预训练统计；新增对齐层、既有融合器/适配器和策略继续训练。保留backbone_trainable开关可进一步解冻。去掉分类头后B2/B3主干约15M/39M，因此不能直接和完整分类器24M/49M参数量混读。

## 实测方法与结果

machine2同一张NVIDIA L4，驱动535.129.03，PyTorch2.7.1+cu126、timm1.0.15、transformers4.56.0。FP32权重+BF16 autocast、eval/inference模式，TF32关闭，不编译。每项30次预热、150次同步计时，顺序测试三种模型。batch表示同时处理的观测数，每份观测含3张图，batch16即48张图。

下表单位ms，使用同步墙钟中位数/p95。完整视觉编码包含共同预处理、主干、通道对齐、多视角融合和适配器；不包含数据读取、解码、BGR转换、CPU→GPU搬运和动作头。显存为该测量进程完整视觉模块及测试张量的peak allocated，不是完整训练显存。

| 模型 | batch | 原生主干 | 主干+对齐 | 完整视觉 | 完整p95 | 显存MiB |
|---|---:|---:|---:|---:|---:|---:|
| dinov3_convnext_base | 1 | 10.012 | 10.013 | 12.631 | 12.735 | 725.2 |
| dinov3_convnext_base | 16 | 157.416 | 158.394 | 170.866 | 172.849 | 1465.8 |
| efficientvit_b2 | 1 | 10.198 | 10.235 | 12.784 | 12.914 | 438.3 |
| efficientvit_b2 | 16 | 69.306 | 69.559 | 81.709 | 82.741 | 987.6 |
| efficientvit_b3 | 1 | 15.247 | 15.307 | 17.853 | 17.971 | 539.2 |
| efficientvit_b3 | 16 | 139.076 | 139.330 | 151.674 | 152.563 | 1215.2 |

测量工具为 `tools/benchmark_visual_backbones.py`，输出 JSON/CSV，包括 CUDA event 和同步墙钟时间、峰值显存、吞吐和形状。原测量共18项，已核对正常退出及完成标记。

## 修改的位置与切换

所有代码直接在开发机 `/data/vepfs/licheng.yang/VFA` 修改。

- `src/vfa/models/policy/vfa/emvis/encoder/efficientvit_encoder.py`：timm MIT B系列包装器、严格加载本地safetensors、空间token格式、坐标和冻结BN。
- `src/vfa/models/policy/vfa/emvis/embodied_vision.py`：新增efficientvit分支及通道对齐，保留DINO分支初始化、参数键和输出。token_forward仍接受已对齐的1024维特征。
- `configs/vfa_train/observation_config/efficientvit_b2_3v.yaml`、`efficientvit_b3_3v.yaml`：完整视觉配置。
- `configs/vfa_train/blocks_ranking_rgb_efficientvit_b2_3v_8gpu_L4.yaml`、B3对应文件：独立实验配置，继承相同训练配方。
- `tools/benchmark_visual_backbones.py`、`tools/download_efficientvit_weights.py`、`tools/check_policy_checkpoint.py`：可复现测量、固定版本下载与严格离线策略重载验证。
- `tests/test_visual_backbones.py`：DINO回归、形状、冻结BN、投影梯度、严格加载和离线重建测试。

通过每次实验的顶层YAML选择完整observation配置；例如defaults中使用 `observation_config/efficientvit_b2_3v`，切B3则改为 `observation_config/efficientvit_b3_3v`，切回Base则用 `observation_config/dinov3cnnl_3v_cnn_b`。不要只改一个backbone_target而保留另一模型的参数/权重目录。

启动B2示例（已有正式训练在跑，不要重复执行）：

```bash
cd /data/vepfs/licheng.yang/VFA
NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 \
  bash scripts/train/train.sh \
  configs/vfa_train/blocks_ranking_rgb_efficientvit_b2_3v_8gpu_L4.yaml
```

Base顶层配置为 `blocks_ranking_rgb_dinov3cnnb_3v_8gpu_L4.yaml`。每次新实验应修改output_dir避免覆盖产物。整个策略checkpoint包含新主干和对齐层，重新加载时load_backbone_pretrain=false，无需再下载原始预训练文件。

## 当前训练配方

四份 blocks_ranking_rgb 顶层配置都设为 `n_obs_steps=1`、`n_image_obs_steps=1`、
`horizon=24`，即当前状态和各视角当前图像，预测未来23帧 joint14 绝对关节动作。
Base 与 B2 使用同样50个 episode、seed42、49训练/1验证、全局batch128、
64个数据worker、200epochs、lr1e-4、BF16和BGR→RGB，视觉主干冻结。

当前输出使用 `MODEL_3v_obs1_h24_8gpu_L4_20261028`，旧训练输出保留。
W&B 离线文件位于相应输出目录的 wandb 子目录。
共享内存帧缓存属于各训练节点，和磁盘 Arrow 缓存、模型权重分别管理。

首轮旧权重（3帧状态/horizon8/未来5动作）在同一验证 episode4 的开环关节
RMSE：Base 0.01566 rad、B2 0.01532 rad。流程见
[开环评测](../scripts/openloop_eval/README.md)。目前暂停100集闭环评测；
没有可用于宣称任务成功率提升的完整结果。自有仿真环境搭建见
[RoboTwin README](../simulation/robotwin2.0/README.md)。
