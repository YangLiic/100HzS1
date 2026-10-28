# VFA RoboTwin 2.0

VFA 自有仿真代码、评测驱动和运行环境。RoboTwin 与 VFA 推理在**同一 Python 进程、
同一环境**中直接调用；默认四步均匀 Euler。三视角 RGB、240×320 输入、
观测帧数和动作长度从 checkpoint 恢复（当前训练配置为 1 帧状态/1 帧图像、horizon24，预测 23 帧动作）。
Aloha 使用 cuRobo 规划、MPLib 执行动作轨迹；无需 PyTorch3D、Isaac Sim 或其他项目。

## 1. 从新机器安装

要求 Ubuntu 22.04、Python 3.10、可用的 NVIDIA GPU/驱动；L4 已验证。
容器需开放 GPU 和 graphics 驱动库。系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv git git-lfs build-essential ffmpeg libgl1 libegl1 \
  libvulkan1 libglib2.0-0 libgomp1 libxrender1 libxext6 libsm6
# 安装 uv：https://docs.astral.sh/uv/getting-started/installation/
cd /path/to/VFA
bash simulation/robotwin2.0/script/setup_env.sh
bash simulation/robotwin2.0/script/setup_curobo.sh
```

cuRobo 编译需要 CUDA Toolkit（nvcc）；默认编译 L4 的 `8.9` 架构，其他显卡设置
`TORCH_CUDA_ARCH_LIST`。源码固定到官方 v0.7.8 对应提交，独立下载到本仓库，
CUDA 扩展使用同一 Torch 编译。

环境创建于 `simulation/robotwin2.0/.venv`，安装带哈希的
`script/requirements.lock`：Torch 2.7.1+cu126、torchvision 0.22.1、
NumPy 1.26.4、SAPIEN 3.0.0b1、MPLib 0.2.1。它与训练环境隔离，
不会修改仓库根目录的 `.venv`。启动脚本自动添加本仓库的 Python 源码路径。
可通过 `PYTHON_BIN`、`ROBOTWIN_ENV`、`UV_CACHE_DIR` 指定解释器、环境和缓存。
只有维护依赖时使用 `setup_env.sh --refresh-lock` 并提交新的锁文件。

## 2. 资产

有共享资产时，只创建软链接，不修改目标内容：

```bash
simulation/robotwin2.0/.venv/bin/python simulation/robotwin2.0/script/link_assets.py \
  --asset-root /DRoboticsAILab/users/licheng.yang/robotwin2.0/assets
```

新机器没有资产时，下载器从 [RoboTwin 官方数据集](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0)
下载并解压三份核心资产，再自动链接：

```bash
bash simulation/robotwin2.0/script/_download_assets.sh /absolute/path/to/robotwin-assets
```

已存在的资产目录会保留；`files` 目录仅为可选展示素材。
本仓库 `task_config/demo_clean.yml` 指定 `motion_planner: curobo`，
加载时覆盖内存中的机器人配置，并在内存中解析资产路径；无需编辑共享或下载的 YAML。

## 3. 检查环境与真实渲染

先确认目标 GPU 空闲，再执行：

```bash
cd simulation/robotwin2.0
source script/headless.sh
.venv/bin/python script/check_env.py --render
cd ../..
```

必须输出正确的环境路径、Torch 版本、`render: true` 和 `[240,320,4]`。
无窗口渲染由 `headless.sh` 设置 NVIDIA EGL/Vulkan；缺驱动库应修复主机或容器配置。

## 4. 单集验证与四卡评测

```bash
bash scripts/eval/eval_robotwin.sh \
  --checkpoint /absolute/path/to/checkpoint-30000 \
  --output /DRoboticsAILab/users/licheng.yang/robotwin2.0/VFA/my_pilot \
  --gpus 0 --episodes 1
# 四卡各25集：改成 --gpus 0 1 2 3 --episodes 100
# 加 --dry-run 只检查配置和种子拆分，不启动 GPU。
```

默认任务 `blocks_ranking_rgb/demo_clean`，默认四步均匀推理；可用 `--task`、
`--task-config`、`--seed-list`、`--nfe` 指定。Base/EfficientViT 根据
checkpoint 的 `config.json` 恢复结构，使用同一入口。
在线 SAPIEN 图像已经是 RGB，不翻通道；保持 JPEG roundtrip 与已有部署一致。
每集每次推理的噪声固定为 `(seed * 1_000_003 + infer_index) % 2**31`。

默认结果根目录为 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/VFA`。
每次使用新输出目录；各卡保存日志、逐集 JSON、视频及环境来源记录。
只有固定种子全部跑完并严格合并后才生成根目录 `result.json`、
`_result.txt` 和 `EVALUATION_COMPLETED`。场景检查失败不会偷偷换种子。
不采集推理延迟。单集完成证明链路可运行，成功率应使用完整固定种子评测。
