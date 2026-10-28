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

## 4. 先用专家筛选场景种子

独立入口不加载 VFA checkpoint，按 seed 递增运行专家。只有
`plan_success` 和任务 `check_success()` 都通过才接受；
每个候选最多3次，取最先通过的100个，不根据 Base/B2 的表现挑选。

```bash
GPU_ID=0 bash scripts/eval/check_expert.sh \
  --task blocks_ranking_rgb --task-config demo_clean \
  --count 100 --start-seed 100000 --max-candidates 2000 \
  --output /absolute/path/to/expert_check_run
# 中断后同一命令加 --resume；代码和场景配置必须保持一致。
```

脚本位于 `script/check_expert.py`，与评测共用 `script/expert_check.py` 的专家判定。
普通规划/任务失败或不稳定场景会尝试下一候选；环境、CUDA 等非场景错误直接报错。
每个候选完成后原子保存 `state.json`，其中含逐次检查结果；
`seeds.partial.txt` 仅供查看进度。凑齐目标数量才生成 `seeds.txt`、
`summary.json` 和 `EXPERT_SEEDS_COMPLETED`，未完成的列表不能用于正式对照。

将完整结果发布为任务的默认列表：

```bash
test -f /absolute/path/to/expert_check_run/EXPERT_SEEDS_COMPLETED
cp /absolute/path/to/expert_check_run/seeds.txt \
  simulation/robotwin2.0/task_config/seedlists/blocks_ranking_rgb_demo_clean.txt
cp /absolute/path/to/expert_check_run/summary.json \
  simulation/robotwin2.0/task_config/seedlists/blocks_ranking_rgb_demo_clean.expert.json
```

Base 与 B2 都使用这同一份列表。默认评测会校验 `.expert.json` 中的任务、
配置、数量和 seed 文件 SHA256，并把它记录到评测 manifest。
评测每集仍会复核专家；固定 seed 检查失败则停止，不替换种子。
其他任务先生成并发布对应的 `任务名_配置名.txt` 和 `.expert.json`。
显式 `--seed-list` 可指定自有列表；无验证摘要时 manifest 标记为未预筛选。

本仓库 `blocks_ranking_rgb/demo_clean` 默认表已于2026-10-28实际检查：
100个候选全部通过，保存100个 seed（100000–100099），编号与原候选表相同，
现在附带专家验证摘要。完整逐候选记录位于
`/DRoboticsAILab/users/licheng.yang/robotwin2.0/VFA/expert_seeds/blocks_ranking_rgb_demo_clean_100_20261028_v1/state.json`。

## 5. 单集验证与四卡评测

统一使用仓库根目录的 `scripts/eval/eval_robotwin.sh`，默认 GPU0–3、
100 个固定种子，每卡25集。执行链路：

| 文件 | 职责 |
|---|---|
| `scripts/eval/eval_robotwin.sh` | 选择仿真环境、设置无窗口运行参数、启动评测 |
| `policy/VFA/evaluate.py`（本平台目录内） | 拆分固定种子，每卡启动一个进程，严格合并结果 |
| `script/eval_policy.py`（本平台目录内） | 单卡逐集仿真，在同一进程直接调用 VFA |

从 VFA 仓库根目录运行：

```bash
# 单卡单集先导；省略 --nfe 时默认4步。
bash scripts/eval/eval_robotwin.sh \
  --checkpoint /absolute/path/to/checkpoint \
  --output /DRoboticsAILab/users/licheng.yang/robotwin2.0/VFA/my_pilot \
  --gpus 0 --episodes 1

# 四卡各25集；此例显式使用10步均匀 Euler。
bash scripts/eval/eval_robotwin.sh \
  --checkpoint /absolute/path/to/checkpoint \
  --output /DRoboticsAILab/users/licheng.yang/robotwin2.0/VFA/my_eval_100 \
  --gpus 0 1 2 3 --episodes 100 --nfe 10
```

加 `--dry-run` 只打印配置、种子分组和输出目录，不启动 GPU 评测。
默认种子文件为 `task_config/seedlists/blocks_ranking_rgb_demo_clean.txt`；
同条件对比应使用相同 checkpoint 选择规则、种子列表和 NFE。

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

### Click Alarmclock 成对评测

任务名为 click_alarmclock，使用 demo_clean。独立专家筛选从 100000 开始，
116 个候选中选出首批 100 个专家同时规划成功、任务成功的种子；
固定列表为 task_config/seedlists/click_alarmclock_demo_clean.txt，
同名 .expert.json 校验任务、数量及 SHA。Base 与 B2 native 共用这份列表。

训练使用 configs/vfa_train/click_alarmclock_{dinov3cnnb,efficientvit_b2_native}_3v_8gpu_L4.yaml，
单帧观测、horizon24、23 个未来绝对关节动作。评测显式传 --task click_alarmclock
--task-config demo_clean --nfe 10 --gpus 0 1 2 3 --episodes 100，分别指定各自最终主模型。
在线 RGB 不翻转，不能用旧 blocks_ranking_rgb 种子列表代替。

无法找到专家接近姿态时，VFA 自有任务脚本标记 plan_success=False 并返回，
由筛选器按专家失败处理；不修改有效场景动作或成功判据。
