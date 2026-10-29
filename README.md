# 100HzS1

基于 Flow Matching 的视觉动作策略，支持 DINOv3 ConvNeXt 与 MIT EfficientViT 编码器。
主要代码在 `src/100hzs1`，训练配置在 `configs/100hzs1_train`；`src/rtpolicy` 保留原始基线。

## 1. 安装训练环境

已验证 Ubuntu 22.04、Python 3.10、NVIDIA L4、PyTorch 2.7.1+cu126。
需要可运行该 PyTorch CUDA 版本的 NVIDIA 驱动，以及 Git、FFmpeg、Python venv 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
git clone https://github.com/YangLiic/100HzS1.git
cd 100HzS1
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv ffmpeg libgl1 libglib2.0-0
bash scripts/setup_train_env.sh
source .venv/bin/activate
python -c "import torch, av, pyarrow, timm; print(torch.__version__, av.__version__, pyarrow.__version__, timm.__version__)"
```

安装脚本使用 `requirements_train.lock` 的版本和文件哈希创建根目录 `.venv`，
然后注册当前仓库。不要在正在使用的环境中重新安装。
`PYTHON_BIN` 可指定 Python 3.10，`S1_TRAIN_ENV` 可指定环境目录；
`UV_CACHE_DIR` 可把下载缓存移到容量充足的磁盘。
只有维护依赖时才修改 `requirements_train.txt` 并运行 `setup_train_env.sh --refresh-lock`，再提交锁文件。

RoboTwin 仿真另有专用环境，见 [仿真搭建说明](simulation/robotwin2.0/README.md)；
RoboTwin 与 100HzS1 推理在该环境的同一进程直接调用，训练环境与仿真环境分别安装。

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

训练代码在 src/100hzs1，配置在 configs/100hzs1_train。项目名为 100HzS1，
Python 包名为 100hzs1；Python 语法不允许直接写数字开头的 import，
外部调用使用 importlib，Hydra 中仍使用 100hzs1 的类路径。
原始 RTPolicy 基线保持独立。

当前 Beat Block Hammer 配方：

| 编码器 | 顶层 YAML（相对 configs/100hzs1_train） |
|---|---|
| DINOv3 ConvNeXt-Base | 100hzs1_beat_block_hammer_dinov3cnnb_3v_8gpu_L4.yaml |
| EfficientViT-B2 native | 100hzs1_beat_block_hammer_efficientvit_b2_native_3v_8gpu_L4.yaml |

两份配置均为单帧状态、每视角单帧图像、三视角、horizon24、200 epochs、
全局 batch128、冻结视觉主干。horizon 包含当前状态条件，因此返回 23 个未来动作。
动作是左右各 6 个关节与 1 个夹爪的 14 维绝对关节位置。

~~~bash
NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 WANDB_MODE=offline \
  bash scripts/train/train.sh \
  configs/100hzs1_train/100hzs1_beat_block_hammer_efficientvit_b2_native_3v_8gpu_L4.yaml \
  --dry-run
~~~

去掉 --dry-run 后正式启动。切换 Base 时换成表中对应 YAML。
在新机器上修改 YAML 中的数据路径，或追加 train_dataset.root=/absolute/task/path。
数据目录需包含 meta/、data/ 和 videos/。新组合先使用独立输出目录，
设置 NUM_PROCESSES=1 GPU_IDS=0，追加 +max_train_steps=2 num_train_epochs=1
train_batch_size=2 dataloader_num_workers=0 验证加载、训练与保存。

训练主配置共享 configs/100hzs1_train/policy_config/dit_fm.yaml；
checkpoint_root 统一设置在这里，默认
/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1，保留任务/配方子目录。
CHECKPOINT_ROOT 或显式 output_dir 可覆盖。每次实验使用独立输出目录。
W&B 默认离线保存在实验目录，按需使用 wandb sync 上传指定 run。

## 4. 默认值与视觉接口

| 项目 | 默认值 | 配置位置 |
|---|---|---|
| 状态帧/每视角图像帧 | 1 / 1 | policy_config/dit_fm.yaml |
| horizon / 未来动作 | 24 / 23 | 同上，n_action_steps 由差值计算 |
| EfficientViT 融合 | native | observation_config/efficientvit_b2_3v.yaml 等 |
| CPU 共享缓存 / 视频预解码 | true / true | dataset_config/demo_unified_dataset.yaml，RoboTwin 配置继承 |
| 离线 BGR→RGB | RoboTwin=true；通用=false | dataset_config/robotwin2_*.yaml |
| 推理步数 / K/V / CUDA Graph / 精度 | 10 / true / true / bf16 | configs/100hzs1_infer/default.yaml |

共享推理 YAML 的唯一源文件为 src/100hzs1/config/inference.yaml；
configs/100hzs1_infer/default.yaml 和训练的 inference/default.yaml 均为相对软链接。
直接推理、开环和闭环采用相同默认值。运行时指定 --inference-config 可选择另一份 YAML，
显式 --nfe、--kv-cache/--no-kv-cache、--cuda-graph/--no-cuda-graph 和 --precision 优先。
修改架构、分辨率或动作长度后需要重新训练，不能仅改推理 YAML 来改变已训练模型的结构。

RoboTwin 离线数据是 BGR，进入模型前转 RGB；在线 SAPIEN 图像已为 RGB，不翻转。
颜色修正不受缓存开关影响。cache_to_ram=false 关闭共享缓存；
true 且 cache_decoded_video=false 只缓存非视频数据、按需解码视频；
两者为 true 时预解码 uint8 帧至训练节点的 CPU 共享内存。Hugging Face 的磁盘
Arrow 缓存是另一层缓存，不是 GPU 显存，也不是 Mac 内存。

三视角原图 240×320，预处理对齐为 256×320。DINOv3 Base 末级通道 1024；
EfficientViT-B2 native 以 384 通道融合和压缩，B3 为 512，末端统一输出
[batch,12,1024]，对接同一策略接口。不同主干由训练 YAML 选择，
推理按 checkpoint 保存的模型配置自动重建，不需手工切换主干。

K/V 缓存仅在同一次动作预测的去噪步骤间复用，下一次观测重新计算。
CUDA Graph 目前只覆盖视觉主干，首次形状/精度组合需要预热和捕获；
训练前向和优化器不受影响。Graph 是默认测试路径，eager 对照需显式选择。
完整策略 checkpoint 包含视觉权重和架构；当前项目不提供旧包名权重兼容层。

~~~python
from importlib import import_module
import torch

load_policy = import_module("100hzs1.models.base_runner").load_policy
policy = load_policy("/absolute/checkpoint", device="cuda:0").eval()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    action = policy.predict_action(obs)["action"]
~~~

## 5. 开环、仿真与验证

- [开环评测](scripts/openloop_eval/README.md)：直接调用策略，离线 BGR→RGB；
  训练/留出验证各一集，每集只存一张 OpenPI 风格 GT/Pred 图，默认输出 openloop_eval。
- [RoboTwin 搭建与评测](simulation/robotwin2.0/README.md)：自有代码与环境，
  可复用资产软链接；仿真与推理同一进程、同一环境，无 TCP/ZMQ 服务。
  四卡入口 scripts/eval/eval_robotwin.sh，默认新输出根为
  /DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1。
  根目录仅 result.json、videos/ 和 logs/；JSON 实时更新成功率及逐集结果。
  新任务首次评测会自动先筛选100个专家成功种子，按任务+场景配置持久保存；
  后续 Base/B2 与其他方法复用同一份，跨进程/机器并发不会重复筛选。
  也可提前运行 scripts/eval/check_expert.sh，具体共享位置与导出方法见平台 README。
- 严格重载主模型/EMA并检查默认 Graph 推理：
  PYTHONPATH=src .venv/bin/python tools/check_policy_checkpoint.py /absolute/checkpoint。
- Graph 测速 tools/benchmark_inference_cuda_graph.py 必须显式传
  --checkpoint NAME=/absolute/checkpoint 和 --dataset /absolute/task；
  默认只测 Graph，--include-eager 才加入 eager。
- CPU 测试：PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v。

代码、YAML、说明和依赖锁文件提交 Git；数据、权重、资产、环境、缓存、
日志、W&B 与评测输出由 .gitignore 排除。
