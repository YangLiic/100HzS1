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
- 两类主干默认冻结；输入三视角 240×320，沿用 VFA 预处理对齐为 256×320。
  EfficientViT 的旧配置使用 `projected` 模式：B2 的 384 通道（B3 为 512）
  先经可训练 Linear 投影到 1024，再融合、压缩。新 `native` 配置自动读取主干通道，
  以 384/512 维融合、压缩，移除前置投影，仅在压缩末尾输出 1024 维。
  两种模式都输出 `[batch,12,1024]`；DINOv3 路径保持不变。
- EfficientViT 代码和共享配置默认 `native`；无模式字段的历史 checkpoint 在加载时补为 `projected`，保留旧结构。继续旧实验须显式设置 projected、dim_2d=1024。
  新实验选择 `blocks_ranking_rgb_efficientvit_b2_native_3v_8gpu_L4.yaml`（B3 同理），
  已配置独立输出目录；窄维融合/压缩权重不兼容旧投影模式，不能原样恢复旧优化器状态。


当前配置已清除未接入计算的历史字段，包括 `model.num_inference_steps`、
`diffusion_step_embed_dim`、`down_dims`、`kernel_size`、`n_groups`、`cond_predict_scale`，
以及调度器中的 `beta_schedule` / `prediction_type`、观察配置中的 `state_key` / `ft_layer_idx`。
实际采样步数只由 `model.noise_scheduler.num_inference_timesteps` 控制（默认10）；
时间频率嵌入仍使用实现默认的256维，未改变网络尺寸。
旧 checkpoint 的历史字段继续接受，无需修改已有权重或配置 JSON。

适配维度和 L4 时延结果见 [编码器说明](docs/efficientvit_backbones.md)。
批量训练视觉编码更快不等于单次策略推理或闭环成功率更好。

PyTorch 推理在 `model.eval()` 后调用 `predict_action(obs)` 时默认启用逐层 K/V 缓存：
同一次动作预测中先计算包含视觉位置编码的 K/V 和 K 归一化，去噪循环内复用，
下一次预测重新生成；缓存不保存到模型或 checkpoint。参数量、权重键、训练计算不变，
历史 checkpoint 无需转换。`predict_action(obs, use_kv_cache=False)` 可关闭缓存作对照。
训练模式或启用梯度时自动使用原计算路径；已有 ONNX 导出包装器继续走未缓存路径。

## 5. 检查与评测

- [开环评测](scripts/openloop_eval/README.md)：直接调用 VFA，无服务器；离线 BGR→RGB，
  输出逐关节 GT / 预测 chunk 曲线、原始预测和误差。
- [RoboTwin 闭环评测](simulation/robotwin2.0/README.md)：统一入口为
  `scripts/eval/eval_robotwin.sh`，默认 GPU0–3 四卡、100个固定种子、每卡25集，
  完整检查后合并结果。先用 `scripts/eval/check_expert.sh` 筛出100个专家成功的
  scene seed，发布为任务默认列表和验证摘要，Base/B2 共用；步骤见平台 README。
  同进程直接推理，默认四步均匀 Euler，`--nfe 10` 可改为十步。
- 保存后严格重载并验证有限动作：`PYTHONPATH=src .venv/bin/python tools/check_policy_checkpoint.py /path/to/checkpoint`。

CPU 测试：

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

源码、YAML、文档和依赖锁文件提交 Git；数据、预训练/训练权重、资产软链接、
环境、缓存、日志、W&B 与评测输出由 `.gitignore` 排除。

### Click Alarmclock 对比

`click_alarmclock_dinov3cnnb_3v_8gpu_L4.yaml` 与 `click_alarmclock_efficientvit_b2_native_3v_8gpu_L4.yaml` 使用相同 50 集数据、单帧观测、horizon24（23 个未来动作）、三视角和冻结主干。离线数据 BGR 转 RGB，在线观测不翻转。两者评测均使用该任务专家筛选的同一 100 scene seed、最终主模型、10 步均匀 Euler。


### 推理 CUDA Graph（默认开启）

安装代码的 PyTorch 策略入口默认使用 `predict_action(obs, use_cuda_graph=True, use_kv_cache=True)`。
`use_cuda_graph` 是每次推理调用的运行时参数，不属于训练配置或 checkpoint；旧权重无需重训或转换。

```python
policy.eval()
# 默认开启：CUDA / eval / no-grad 时捕获并重放视觉主干。
result = policy.predict_action(obs)
# 对照测试或调试时显式关闭。
result = policy.predict_action(obs, use_cuda_graph=False)
```

本实现仅捕获视觉主干；图像预处理、投影、视角融合、压缩、DiT 采样保持原有执行方式，
K/V 缓存开关独立。这样无需更改训练 pipeline、loss 或优化器，也不会固定采样的随机噪声。
`compute_loss` 和普通编码器调用仍默认 eager，CPU、train 和启用梯度的编码器也保持 eager。
ONNX / TensorRT 推理不由这个 PyTorch 开关控制。

首次遇到输入形状/精度组合时会预热并捕获，首次调用耗时及显存占用会增加。
每个视觉编码器最多保留一个图；切换 batch、尺寸、精度或 CUDA stream 时重新捕获。
捕获前会检查主干中的常见CPU同步/动态输出操作；准备阶段失败会记录 warning 并为当前签名回退到 eager，不会每帧重复尝试。
若CUDA捕获已开始后发生错误，会明确报错并要求以 `use_cuda_graph=False` 重启推理，避免当前PyTorch版本的捕获失败污染后续随机采样。
进入训练、移动/转换模型或加载 state_dict 时自动清空图；也可调用
`policy.obs_encoder.scene_encoder.clear_cuda_graph_cache()` 主动释放。
直接替换视觉主干内部子模块或修改其执行逻辑后，也应主动清空缓存。

离线评测脚本 `scripts/openloop_eval/evaluate.py` 默认启用，可加 `--no-cuda-graph` 关闭。
旧的 eager 性能复测脚本已显式传 `use_cuda_graph=False`，保持既有测试口径。
本次验证及完整策略时延见 [推理 CUDA Graph 报告](docs/VFA/inference_cuda_graph_20261028.md)。

### 推理默认与测速顺序

PyTorch 的 predict_action 默认 use_cuda_graph=True、use_kv_cache=True。
CUDA Graph 当前只覆盖视觉主干；训练前向和优化器不受影响。
测速入口 tools/benchmark_inference_cuda_graph.py 默认只运行 Graph，不附带 eager 对照；
仅在明确选择后添加 --include-eager。开环默认同样开启 Graph。
闭环记录实际 capture/replay，要求使用 Graph 时不能把未捕获的结果记为 Graph 结果。
