# VFA 直接推理接口

`deploy_policy.py` 实现 RoboTwin 的 `get_model / eval / reset_model`。
仿真与模型使用 `simulation/robotwin2.0/.venv`，同进程直接调用。
`evaluate.py` 启动每卡一个进程并严格合并固定种子结果。
安装、资产和运行命令见 [平台 README](../../README.md)。


统一从仓库根目录运行 `bash scripts/eval/eval_robotwin.sh --checkpoint /path/to/checkpoint`。
默认四卡 GPU0–3、100集、每卡25集；`--nfe` 控制推理步数，
`--dry-run` 检查种子拆分。平台的 `script/eval_policy.py` 由调度器启动。
