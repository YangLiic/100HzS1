# 100HzS1 同进程策略

使用根目录 scripts/eval/eval_robotwin.sh，不直接启动内部 worker。
evaluate.py 负责四卡调度和实时 result.json，script/eval_policy.py 负责每卡仿真，
deploy_policy.py 在同一 Python/Torch 环境直接调用 checkpoint 指定的主干与策略。
不使用 TCP/ZMQ。

默认10步均匀 Euler、KV与视觉 CUDA Graph开启，统一读取
configs/100hzs1_infer/default.yaml。在线图像已是 RGB，不进行 BGR 转换。
结果结构、实时字段与种子筛选见平台 README。
