# 开环 GT 对比

直接调用 100HzS1，不使用仿真或推理服务器。离线 LeRobot 图像默认 BGR→RGB；
推理默认10步均匀 Euler、K/V缓存与视觉 CUDA Graph开启，BF16。
共享开关见 configs/100hzs1_infer/default.yaml。

~~~bash
bash scripts/openloop_eval/run.sh \
  --checkpoint /absolute/path/to/task/checkpoint \
  --task beat_block_hammer
~~~

默认读取该训练输出的 resolved_config.yaml，复用训练数据集的筛选、随机划分和采样逻辑，
选训练集一集、留出验证集一集。验证集在文件名中标为 test，并非额外独立测试数据。
当前 Hammer 的选择是 train episode0、test episode4；可用 --episodes 指定，
程序仍会检查归属。缺少训练配置时传 --training-config /absolute/resolved_config.yaml。

图像默认直接保存在 /DRoboticsAILab/users/licheng.yang/openloop_eval：

~~~text
<权重目录名>_train_ep000000.png
<权重目录名>_test_ep000004.png
~~~

checkpoint-10000 这类目录会自动加上所属训练目录名，避免不同模型重名。
每个 episode 只有一张逐关节 GT/Pred 对比图，保持指定 OpenPI 原绘图样式；
不生成 NPZ、指标 JSON、完成标记或 episode 子目录。指标和完成信息仅输出到终端。
已有同名图片时拒绝覆盖，可换 --output 目录。

观测使用当前 GT 状态/图像，各窗口独立调用模型；GT 是随后完整 chunk 的专家动作。
不把预测反馈成下一窗口输入，尾部不足完整 chunk 的帧不绘制。
开环误差不是闭环成功率。

常用参数：--dataset、--episodes、--anchors、--stride、--output、--source-color bgr/rgb、
--training-config、--inference-config、--nfe、--precision、--cuda-graph/--no-cuda-graph、
--kv-cache/--no-kv-cache。--dry-run 仅显示实际划分、图片路径和配置，不运行 GPU。
