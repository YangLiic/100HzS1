# 100HzS1 RoboTwin 2.0

100HzS1 自有仿真代码、评测驱动和运行环境。RoboTwin 与 100HzS1 推理在**同一 Python 进程、
同一环境**中直接调用；默认十步均匀 Euler。三视角 RGB、240×320 输入、
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
cd /path/to/100HzS1
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

独立入口不加载 100HzS1 checkpoint，按 seed 递增运行专家。只有
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
`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/expert_seeds/blocks_ranking_rgb_demo_clean_100_20261028_v1/state.json`。

## 5. 四卡评测与实时结果

统一入口 scripts/eval/eval_robotwin.sh。默认 GPU0–3，每卡25集，共100个已通过专家的固定 seed；
模型从 checkpoint 配置自动重建主干。10步均匀 Euler、K/V缓存、视觉 CUDA Graph、
BF16 默认值统一来自 configs/100hzs1_infer/default.yaml。在线 SAPIEN RGB 不翻转。

~~~bash
bash scripts/eval/eval_robotwin.sh \
  --checkpoint /absolute/path/to/checkpoint \
  --task beat_block_hammer --gpus 0 1 2 3 --episodes 100
~~~

单集验证改成 --gpus 0 --episodes 1。--dry-run 只显示配置与分组。
默认 seedlist 为 task_config/seedlists/<task>_<profile>.txt，必须配套专家验证摘要；
Base/B2 使用同一份列表。--seed-list 可显式指定另一份固定列表。

默认输出：

~~~text
/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/<task>/<权重目录名>/<时间>/
├── result.json
├── videos/seed_<seed>.mp4
└── logs/
    ├── gpu0.log ... gpu3.log
    └── gpu0.jsonl ... gpu3.jsonl
~~~

根目录只有一个结果 JSON、视频目录和日志目录，不再生成各卡结果子树、
重复 TXT、manifest、seed 副本或完成标记。配置、种子及 SHA、运行环境、代码摘要
直接放进 result.json；journal 保存在 logs，供多进程汇总与故障排查。

result.json 在运行中约每秒原子更新，随时可打开：

| 字段 | 含义 |
|---|---|
| status | running / completed / failed / interrupted |
| planned_episodes / completed_episodes | 计划数 / 已完成数 |
| successes / failures | 已完成集数中的成功 / 失败数 |
| success_rate | successes / completed_episodes；尚未完成任何一集时为 null |
| results | 所有已完成 episode 的 seed、GPU、成功与否、动作数、推理次数和视频 |
| workers | 每卡当前 seed、完成数、状态及实际 Graph 捕获/重放次数 |

只有各 worker 成功退出、固定种子覆盖完整且视频存在，才写 status=completed。
失败或中断会保留已完成结果；运行中成功率不应当作最终100集成功率。
不记录单次推理延迟。

实时查看简洁汇总和最近 episode：

~~~bash
python3 scripts/eval/watch_robotwin.py /absolute/evaluation/result.json
# 只打印一次
python3 scripts/eval/watch_robotwin.py /absolute/evaluation/result.json --once
~~~

## 6. Beat Block Hammer

beat_block_hammer_demo_clean 的100个固定种子已由独立专家筛选：
118个候选，18个拒绝。列表及验证摘要提交在 task_config/seedlists。
正式比较先用最终主模型跑训练/留出验证各一集开环，再使用相同100个种子四卡闭环。
训练和推理均不修改其他项目的代码；资产可使用共享目录的软链接。
