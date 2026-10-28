# VFA

基于 Flow Matching 的视觉动作策略，支持 DINOv3 ConvNeXt 与 MIT EfficientViT 编码器。
主要代码在 `src/vfa`，训练配置在 `configs/vfa_train`；`src/rtpolicy` 保留原始基线。

## 1. 安装训练环境

已验证 Ubuntu 22.04、Python 3.10、NVIDIA L4、PyTorch 2.7.1+cu126。
需要可运行该 PyTorch CUDA 版本的 NVIDIA 驱动，以及 Git、FFmpeg、Python venv 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
git clone https://github.com/YangLiic/VFA.git
cd VFA
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv ffmpeg libgl1 libglib2.0-0
bash scripts/setup_train_env.sh
source .venv/bin/activate
python -c "import torch, av, pyarrow, timm; print(torch.__version__, av.__version__, pyarrow.__version__, timm.__version__)"
```

安装脚本使用 `requirements_train.lock` 的版本和文件哈希创建根目录 `.venv`，
然后注册当前仓库。不要在正在使用的环境中重新安装。
`PYTHON_BIN` 可指定 Python 3.10，`VFA_TRAIN_ENV` 可指定环境目录；
`UV_CACHE_DIR` 可把下载缓存移到容量充足的磁盘。
只有维护依赖时才修改 `requirements_train.txt` 并运行 `setup_train_env.sh --refresh-lock`，再提交锁文件。

RoboTwin 仿真另有专用环境，见 [仿真搭建说明](simulation/robotwin2.0/README.md)；
RoboTwin 与 VFA 推理在该环境的同一进程直接调用，训练环境与仿真环境分别安装。

## 2. 下载视觉预训练权重

以下命令用于新机器；已有有效目录或共享权重软链接时可以直接使用。
权重统一放在 `pretrained/`，不提交 Git。下载前关闭 Hugging Face 离线模式：

```bash
unset HF_HUB_OFFLINE
export HF_HOME="$PWD/.cache/huggingface"
mkdir -p pretrained
```

### DINOv3 ConvNeXt-Base

来源：[Meta 官方模型](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m)。
先在模型页面申请访问、接受使用条款，再使用有权限的 Hugging Face 账号登录：

```bash
.venv/bin/hf auth login
.venv/bin/hf download facebook/dinov3-convnext-base-pretrain-lvd1689m \
  --local-dir pretrained/dinov3-cnn-base \
  --include config.json preprocessor_config.json model.safetensors README.md
```

目录必须包含模型、`config.json` 与 `preprocessor_config.json`。
当前 Base 的各阶段通道为 `[128,256,512,1024]`，不是 ConvNeXt-Large。

### EfficientViT-B2

使用 [MIT EfficientViT](https://github.com/mit-han-lab/efficientvit) 的
[timm ImageNet-1K 预训练分类权重](https://huggingface.co/timm/efficientvit_b2.r256_in1k)。
下载器记录来源、实际版本、SHA256 和文件大小；下面固定到本次实验使用的版本：

```bash
.venv/bin/python tools/download_efficientvit_weights.py \
  --root pretrained --models efficientvit_b2.r256_in1k \
  --revision a55d5b6bf8c67a8496c73ab2d6013c642846b572
```

目录结构：

```text
pretrained/
├── dinov3-cnn-base/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors
└── efficientvit_b2.r256_in1k/
    ├── config.json
    ├── model.safetensors
    └── provenance.json
```

可选 B3：下载器改成 `--models efficientvit_b3.r256_in1k`，省略 B2 的 `--revision`。
可选 DINOv3 Large：官方仓库名改成
`facebook/dinov3-convnext-large-pretrain-lvd1689m`，目录用 `pretrained/dinov3-cnn-large`。

完整策略 checkpoint 已包含视觉主干和策略权重，恢复推理无需再次下载视觉预训练文件。

## 3. 配置与训练

| 编码器 | 顶层配置（相对 configs/vfa_train） |
|---|---|
| DINOv3 ConvNeXt-Base | `blocks_ranking_rgb_dinov3cnnb_3v_8gpu_L4.yaml` |
| DINOv3 ConvNeXt-Large | `blocks_ranking_rgb_dinov3cnnl_3v_8gpu_L4.yaml` |
| EfficientViT-B2 | `blocks_ranking_rgb_efficientvit_b2_3v_8gpu_L4.yaml` |
| EfficientViT-B3 | `blocks_ranking_rgb_efficientvit_b3_3v_8gpu_L4.yaml` |

这四份配置当前都使用单帧观测、`horizon=24`：

| 参数 | 含义 | 当前值 |
|---|---|---:|
| `n_obs_steps` | 输入的关节状态时间帧数 | 1 |
| `n_image_obs_steps` | 每个视角输入的图像时间帧数，与视角数量无关 | 1 |
| `horizon` | 状态条件与未来动作组成的总序列长度 | 24 |
| `n_action_steps` | 推理返回的未来动作数，默认 horizon - n_obs_steps | 23 |

例如 `n_obs_steps=3, n_image_obs_steps=1` 表示 3 帧状态、各视角仅最新 1 帧图像。
当前 `1/1` 表示当前状态与左/右/头三视角的当前图像，没有历史帧。
模型始终预测 14 维绝对关节动作：左右各 6 个关节和 1 个夹爪。

配置中的数据与输出路径是现有开发机路径。新机器用命令行覆盖，
或编辑原 YAML；`train_dataset.root` 必须指向含 `meta/`、`data/`、`videos/` 的单任务目录。
每次实验使用新的输出目录。Base 示例：

```bash
NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 WANDB_MODE=offline \
  bash scripts/train/train.sh \
  configs/vfa_train/blocks_ranking_rgb_dinov3cnnb_3v_8gpu_L4.yaml \
  train_dataset.root=/absolute/path/to/robotwin2.0_lerobot_clean_joint14/blocks_ranking_rgb \
  model.obs_encoder.emvis_config.backbone_path="$PWD/pretrained/dinov3-cnn-base" \
  output_dir=/absolute/path/to/checkpoints/VFA/base_obs1_h24
```

训练 B2 时改成 B2 顶层 YAML，并把 `backbone_path` 改为
`"$PWD/pretrained/efficientvit_b2.r256_in1k"`。B3、Large 同理，需要成套切换
observation 配置、模型规格和权重路径，不能只改 `backbone_target`。

先给命令加 `--dry-run` 检查启动参数；单卡调试使用 `NUM_PROCESSES=1 GPU_IDS=0`。
新环境/新模型应先用 `+max_train_steps=2 num_train_epochs=1 dataloader_num_workers=0`
和独立 smoke 输出目录验证，再运行正式训练。

默认 checkpoint 根为 `/DRoboticsAILab/users/licheng.yang/checkpoints/VFA`；
`CHECKPOINT_ROOT` 可整体覆盖。W&B 默认离线，建议设置 `WANDB_DIR` 到本次输出目录；
保留本地记录后，可按需要使用 `wandb sync` 上传指定 run。

## 4. 图像、缓存与编码器接口

- 此 RoboTwin 离线数据虽然目录名含 `_rgb`，实际通道为 BGR；
  RoboTwin dataset 配置启用 `train_dataset.bgr_to_rgb=true`，进入模型前转成 RGB。
  通用数据默认关闭此开关，真正 RGB 数据应显式设 false。在线 SAPIEN 图像已经是 RGB，不翻转。
- `train_dataset.cache_to_ram=true` 是总开关。设为 false 时，
  `cache_decoded_video` 不生效，恢复按常规路径读取/解码；
  设为 true、`cache_decoded_video=false` 时仅缓存非视频数据，视频按需解码；
  两者为 true 时还会预解码视频为 uint8 帧，放入训练节点的 CPU 共享内存。
  不占 GPU 显存，也不占其他开发机内存；Hugging Face 的磁盘 Arrow 缓存是另一层缓存。
- 两类主干默认冻结；训练融合器、适配器、策略，以及 EfficientViT 的通道对齐层。
  输入三视角 240×320，沿用 VFA 预处理对齐为 256×320。
  B2 的 384 通道（B3 为 512）投影到 1024，再经原融合与适配模块输出
  `[batch,12,1024]`，与 Base 策略接口相同。

适配维度和 L4 时延结果见 [编码器说明](docs/efficientvit_backbones.md)。
批量训练视觉编码更快不等于单次策略推理或闭环成功率更好。

## 5. 检查与评测

- [开环评测](scripts/openloop_eval/README.md)：直接调用 VFA，无服务器；离线 BGR→RGB，
  输出逐关节 GT / 预测 chunk 曲线、原始预测和误差。
- [RoboTwin 闭环评测](simulation/robotwin2.0/README.md)：统一入口为
  `scripts/eval/eval_robotwin.sh`，默认 GPU0–3 四卡、100个固定种子、每卡25集，
  完整检查后合并结果。同进程直接推理，默认四步均匀 Euler，`--nfe 10` 可改为十步。
- 保存后严格重载并验证有限动作：`PYTHONPATH=src .venv/bin/python tools/check_policy_checkpoint.py /path/to/checkpoint`。

CPU 测试：

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

源码、YAML、文档和依赖锁文件提交 Git；数据、预训练/训练权重、资产软链接、
环境、缓存、日志、W&B 与评测输出由 `.gitignore` 排除。
