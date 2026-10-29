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

## 4. 一次筛选，所有模型共用

默认按 task + task-config 锁定一份100个专家成功的 scene seed，独立于
模型、backbone、checkpoint、采样步数和输出目录。新任务第一次评测会先自动调用
check expert；已有列表直接复用，不为 Base/B2 分别选种子。

也可提前准备：

~~~bash
bash scripts/eval/check_expert.sh --task adjust_bottle
~~~

默认共享目录为
/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/seed_registry/<task>/<profile>/set/，
其中 seeds.txt、summary.json 写完后，原子发布 COMMITTED.json 标记。S1_SEED_ROOT 或 --seed-root
可指定其他共享位置；比较方法应使用同一位置。

- 只有专家规划与任务成功均通过才接受 seed；发布时严格检查100个唯一整数和 SHA256。
- 多机器同时准备同一任务时只有一个筛选者，其余等待并复用同一结果。
- start-seed、attempts 等筛选参数只用于首次创建；已有100种子不会因此变更。
- 已发布列表不被自动重选、追加或覆盖；核心场景、机器人和相机代码/配置的摘要变化会明确拒绝混用。
  跨机器比较还应保持相同资产与仿真依赖版本。
- 仓库已有的验证列表会被原样导入共享目录，保持原来的种子和顺序，不重新筛选。
- 每次模型 rollout 仍要独立执行。共享的是种子/场景定义，不是其他模型的结果；
  物理仿真不承诺逐位一致，复检瞬时失败只重试原 seed，不偷偷换成别的 seed。
- --dry-run 不筛选、不写共享目录、不启动 GPU。--seed-list 是显式自定义覆盖，结果会记录其 SHA。

要把新任务的固定列表随 Git 分发到另一台没有共享盘的机器：

~~~bash
bash scripts/eval/check_expert.sh --task adjust_bottle --publish
~~~

这会导出 task_config/seedlists/<task>_<profile>.txt 和 .expert.json；
只复用已锁定列表，拒绝覆盖成另一套 seed。随后提交这两份文件。
--output 可另外保存筛选诊断；小于100集的检查必须显式 --output，不会注册为公共100集列表。
筛选中断会保留 build/state.json，正常重试从已处理候选后继续；异常退出遗留的锁会明确报错，
不会为了继续而生成一份不同列表。

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
默认先准备/读取上一节的共享种子注册表；仓库随附的专家列表会原样导入。
Base/B2 使用同一份100种子。--seed-list 可显式指定另一份列表，其 SHA 会记录在结果中。

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

专家复检的暂时失败会保留同一 seed，在独立 worker 进程中最多重试3次；
已完成 episode 不重跑，其他卡继续推进。其他错误仍明确终止。
中断后可在新目录继续：追加 --resume-from /absolute/failed_run --output /absolute/new_run。
程序校验权重、固定种子、NFE、Graph/KV/精度和颜色设置一致，复制已完成记录与视频，
只运行未完成的 seed；旧结果原样保留。--max-worker-restarts 可调整重试上限。

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
