# 100HzS1 开环检查

直接加载 100HzS1 checkpoint，读取 LeRobot parquet 和 MP4，用真实观测预测下一段动作（当前默认单帧，无历史）并与 GT 比较。无 RoboTwin、ZMQ 或服务端依赖。参考 OpenPI 的逐关节绘图方式，实际推理调用 100HzS1 的 `predict_action`。

## 运行

在 100HzS1 根目录，使用训练环境（已有 Torch、PyAV、PyArrow）：

```bash
uv pip install --python .venv/bin/python -r scripts/openloop_eval/requirements.txt
scripts/openloop_eval/run.sh \
  --checkpoint /DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/blocks_ranking_rgb/dinov3cnnb_3v_8gpu_L4_20261028/checkpoint-30000 \
  --episodes 0 4 --nfe 10 \
  --output /DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/openloop/base_ckpt30000
```

B2 只替换 checkpoint 路径，输出必须使用新目录。默认数据父目录是 `/DRoboticsAILab/vepfs_20260622/licheng.yang/datasets/robotwin2.0_lerobot_clean_joint14`，任务 `blocks_ranking_rgb`；可通过 `--dataset` 指定单任务目录或父目录，通过 `--task` 选择任务。默认完整评测 episode 0、4（原训练的 4 是验证集），步幅为模型动作 chunk 长度；`--anchors 30 120 220` 只检查指定窗口。

## 数据与时间对齐

- 离线数据默认 `--source-color bgr`，视频解码后 **只交换一次 B/R**，转 RGB、CHW、float/255，分辨率由 checkpoint 的原编码器处理。真正 RGB 数据用 `--source-color rgb`。闭环 SAPIEN 图像本来是 RGB，不能套用此翻转。
- 对第一帧未来动作 a：状态输入 `[a-n_obs,a)`，图像输入 `[a-n_image,a)`，GT 是 `action[a:a+chunk]`。当前配置为 1 帧状态、1 帧图像、23 帧未来动作（horizon24）；图像时刻 a-1。跳过边界不足的窗口，无补齐。
- 14 维绝对关节动作：左右各 6 个关节（弧度）+1 个夹爪。图和误差使用反归一化后的原单位。
- 默认 BF16 autocast，与直接 RoboTwin 推理相同；`--precision fp32` 会显式转换全部权重和时间嵌入的 dtype，避免混合 dtype 冲突。默认不额外 JPEG 编解码，原数据已经是视频。
- 每个窗口都重新使用 GT 观测，**不把预测作为下一窗口输入**。固定噪声种子按 episode/anchor 生成；同条件可复现。NFE 是 100HzS1 原生均匀 Euler，并检查实际网络调用次数。

## 产物与判读

`summary.json`：每集及汇总的关节 RMSE、夹爪 MAE、逐预测步误差；包含保持当前状态/匀速外推基线，以及运动窗口统计。`--ablations` 对每集 4 个窗口额外比较黑图、错配图像、未转色、JPEG、FP32，使用相同噪声。

每集只生成一张 state_comparison.png：14 维关节/夹爪的 GT 与 Pred 轨迹。
绘图代码逐句沿用 OpenPI inference/openloop/openloop_eval.py 中
save_state_comparison 的第一部分：4 列布局、蓝实线 GT、红虚线 Pred、
灰色 chunk 分界线、字体/图例/网格/150dpi 均保留。不生成 MAE 图、chunk 局部图或输入相机图。
来源与 SHA 见 plotting_reference.json，未来 GT 对齐和推理数值没有改变。

数值产物仍为 predictions.npz、metrics.json 和 summary.json；完成标记 OPENLOOP_COMPLETED，
失败不写标记。CUDA Graph 默认开启，只捕获视觉主干；不自动追加 eager 对照。

短 chunk 下保持状态也可能误差很低，要结合运动阶段、夹爪切换和基线判读。开环正确仅能帮助排除数据与推理问题，不等于闭环成功，也不能单独证明 chunk 长度是失败原因。

新一轮单帧配置为 `n_obs_steps=1, n_image_obs_steps=1, horizon=24`，未来动作 chunk=23（总窗口 24 含 1 帧观测）。脚本自动读取 checkpoint 参数。仅 1 帧状态时无法估计速度，匀速基线退化为保持状态，summary 的 `constant_velocity_available=false` 标明这一点。


## 本次旧权重检查

两份 checkpoint-30000，NFE10，episode 0（训练）与 episode 4（验证），每集 95 个完整窗口：

| 编码器 | ep0 关节 RMSE/rad | ep4 关节 RMSE/rad | ep4 夹爪 MAE |
|---|---:|---:|---:|
| DINOv3 ConvNeXt-Base | 0.00457 | 0.01566 | 0.00421 |
| EfficientViT-B2 | 0.00476 | 0.01532 | 0.00405 |

ep4 保持状态/匀速外推基线的关节 RMSE 为 0.06788/0.02175 rad。两模型曲线可贴合 GT；8 个抽样窗口中黑图、错配图像明显变差，FP32/JPEG 影响较小。这里只核查离线输入和短期预测，未证明闭环失败的唯一原因。

完整结果：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/openloop/{base_checkpoint30000_v2,efficientvit_b2_checkpoint30000_v2}`。v1 是未完成的 FP32 诊断尝试，不作为有效结果。后续新权重使用新输出目录，按模型自身 n_obs/horizon 读取。

运行时默认读取 configs/100hzs1_infer/default.yaml（10步、KV=true、CUDA Graph=true、bf16）；
--inference-config 可替换 YAML，显式 --nfe/--kv-cache/--no-kv-cache/--cuda-graph/--no-cuda-graph 优先。
