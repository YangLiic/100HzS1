# beat_block_hammer：16 个策略的 TensorRT FP16 延迟与参数量

## 测量口径

- NVIDIA L4，batch=1，单个样本包含 3 路 RGB（原始 240×320）及 1 个 state；TensorRT 10.9.0.34，KV cache 开启，CUDA Graph 关闭。
- FP16 表示 backbone、encoder 和 step 引擎启用 TensorRT FP16 builder flag，允许混合精度；后处理引擎保留 FP32。
- 视觉编码：三视图经训练一致的预处理后，仅计 backbone TensorRT 引擎；不含图像预处理、视角融合和压缩。完整推理：GPU 上原始 RGB、state、预生成噪声进入，包含预处理、视觉 backbone、融合、压缩、条件投影、KV 构造、DiT 各步和动作后处理。
- 延迟为 CUDA event 计时的单次同步 p50（ms）；每模型每步数预热 30 次、采样 3×100 次。输入已在 GPU，不含 H2D/D2H、采图、视频解码、网络传输、噪声生成、构建、加载及预热；不是端到端机器人控制延迟。
- 参数按独立 PyTorch 参数对象计数，单位 M；动作部分 = DiT（含时间编码、AdaRMS 和 action-time fusion）+ state_adaptor。非视觉总计另包含 scene_adaptor；不包含 obs_encoder（视觉 backbone、视角融合和视觉压缩），也不计 normalizer 统计。
- 每行是各自训练后的权重，不是简单地对同一个权重替换时间注入；参数量是架构计数，FP16 并不会改变参数个数。

## 结果

| Backbone | 方法 | 注入 | 视觉 backbone (ms) | 完整 1-step (ms) | 完整 4-step (ms) | DiT + state (M) | scene adaptor (M) | 非视觉总计 (M) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| B0 | FM | tau_token | 0.378 | 4.030 | 12.302 | 353.208 | 6.296 | 359.504 |
| B0 | FM | tau_AdaRMS | 0.377 | 6.493 | 22.119 | 676.323 | 6.296 | 682.619 |
| B0 | MeanFlow | tau_token | 0.375 | 4.083 | — | 357.931 | 6.296 | 364.227 |
| B0 | MeanFlow | tau_AdaRMS | 0.376 | 6.510 | — | 681.046 | 6.296 | 687.342 |
| B1 | FM | tau_token | 0.767 | 4.420 | 12.563 | 353.208 | 6.296 | 359.504 |
| B1 | FM | tau_AdaRMS | 0.780 | 6.869 | 22.290 | 676.323 | 6.296 | 682.619 |
| B1 | MeanFlow | tau_token | 0.767 | 4.520 | — | 357.931 | 6.296 | 364.227 |
| B1 | MeanFlow | tau_AdaRMS | 0.768 | 6.933 | — | 681.046 | 6.296 | 687.342 |
| B2 | FM | tau_token | 1.578 | 5.221 | 13.491 | 353.208 | 6.296 | 359.504 |
| B2 | FM | tau_AdaRMS | 1.554 | 7.603 | 23.106 | 676.323 | 6.296 | 682.619 |
| B2 | MeanFlow | tau_token | 1.563 | 5.225 | — | 357.931 | 6.296 | 364.227 |
| B2 | MeanFlow | tau_AdaRMS | 1.565 | 7.675 | — | 681.046 | 6.296 | 687.342 |
| DINOv3 Base | FM | tau_token | 5.715 | 9.684 | 17.762 | 353.208 | 6.296 | 359.504 |
| DINOv3 Base | FM | tau_AdaRMS | 5.721 | 11.994 | 27.149 | 676.323 | 6.296 | 682.619 |
| DINOv3 Base | MeanFlow | tau_token | 5.801 | 9.798 | — | 357.931 | 6.296 | 364.227 |
| DINOv3 Base | MeanFlow | tau_AdaRMS | 5.761 | 12.020 | — | 681.046 | 6.296 | 687.342 |

## 参数量对比

- 在本次 2048 宽、8 层动作网络中，AdaRMS 相比 token 的 DiT + state 参数均增加约 **323.115M**；FM 从 353.208M 增至 676.323M，MeanFlow 从 357.931M 增至 681.046M。两组差值由各自实际 checkpoint 统计，包含动作时间融合、逐层调制与门控等新增参数。
- 每组使用各自训练的一份权重和固定的合成输入；延迟反映本次引擎与硬件设置，不能单凭这些结果保证跨硬件的相同加速比。

## 数值校验与产物

- 每个模型与步数均使用相同生成的 RGB、state 和噪声，对比 TensorRT 输出与 PyTorch FP32 eager 输出；要求全部有限、RMSE ≤ 0.01、最大绝对误差 ≤ 0.05。该检查用于发现导出偏差，不等于真实机器人任务精度验证。
- 引擎构建时 TensorRT 对部分 FP16 归一化给出潜在溢出提示；本报告保留 FP32 后处理，且以上校验全部通过，但不能保证所有输入都无数值风险。
- 原始计划、逐次时延、校验数据、参数明细和日志：`/DRoboticsAILab/users/licheng.yang/quantization/beat_hammer_trt16_20261030_v1`。模型权重及各次闭环评测来源见该目录 `cases_input.json`。
- 先前 100 episode 闭环结果基于 PyTorch BF16；不能视为本次 TensorRT FP16 的闭环成功率。
