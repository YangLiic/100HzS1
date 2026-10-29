# TensorRT FP16 / INT8 延迟对比（2026-10-29）

NVIDIA L4 23GB / 72W / 默认动态时钟；TensorRT 10.9.0.34；TF32 关闭。以下为 **p50 / p95，ms/整批**；BS1=3 视角，BS16=48 张 RGB，不除以 16。固定输入，每项 3 轮、每轮预热 30 次后同步计时 100 次；CG=CUDA Graph。

## 独立主干编码

输入为预处理后的 RGB（每视角 3×256×320），仅计主干特征提取。

| 模型 | 精度 | BS | TensorRT | TensorRT + CG |
|---|---|---:|---:|---:|
| DINOv3 ConvNeXt-B | FP16 | 1 | 5.79 / 5.85 | 5.58 / 5.61 |
| DINOv3 ConvNeXt-B | FP16 | 16 | 120.31 / 123.52 | 120.32 / 123.27 |
| DINOv3 ConvNeXt-B | INT8 | 1 | 7.62 / 7.71 | 7.42 / 7.56 |
| DINOv3 ConvNeXt-B | INT8 | 16 | 127.67 / 129.63 | 127.28 / 129.42 |
| EfficientViT-B2 native | FP16 | 1 | 1.62 / 1.64 | 1.48 / 1.49 |
| EfficientViT-B2 native | FP16 | 16 | 29.67 / 31.19 | 29.58 / 31.08 |
| EfficientViT-B2 native | INT8 | 1 | 2.92 / 2.94 | 2.73 / 2.74 |
| EfficientViT-B2 native | INT8 | 16 | 52.08 / 53.57 | 52.06 / 53.52 |

## 完整推理管道

GPU 原始 RGB（240×320）+ state + 固定噪声 → 23×14 动作；包含预处理、主干、融合/压缩、scene adaptor、一次 K/V 计算、FM 采样、反归一化。

| 模型 | 精度 | BS | 4步 TRT | 4步 TRT+CG | 10步 TRT | 10步 TRT+CG |
|---|---|---:|---:|---:|---:|---:|
| DINOv3 ConvNeXt-B | FP16 | 1 | 17.88 / 18.22 | 17.21 / 17.53 | 34.26 / 34.56 | 33.02 / 33.32 |
| DINOv3 ConvNeXt-B | FP16 | 16 | 155.02 / 158.10 | 154.66 / 157.63 | 197.47 / 201.37 | 196.80 / 200.99 |
| DINOv3 ConvNeXt-B | INT8 | 1 | 14.98 / 15.31 | 14.28 / 14.68 | 24.75 / 25.16 | 23.40 / 23.81 |
| DINOv3 ConvNeXt-B | INT8 | 16 | 148.52 / 150.78 | 146.68 / 150.13 | 172.36 / 175.80 | 170.49 / 174.52 |
| EfficientViT-B2 native | FP16 | 1 | 13.25 / 13.40 | 12.65 / 12.78 | 29.44 / 29.56 | 28.29 / 28.37 |
| EfficientViT-B2 native | FP16 | 16 | 58.89 / 61.26 | 58.60 / 61.15 | 99.76 / 102.45 | 99.55 / 102.80 |
| EfficientViT-B2 native | INT8 | 1 | 10.22 / 10.40 | 9.46 / 9.68 | 20.16 / 20.34 | 18.78 / 18.88 |
| EfficientViT-B2 native | INT8 | 16 | 70.81 / 72.28 | 69.83 / 71.19 | 95.25 / 96.88 | 93.78 / 95.56 |

## 量化核验

显式 Q/DQ PTQ（A8 按张量、W8 按通道，Entropy2）：128 组真实观测（episode 0–7），FM 单步校准 512 样本覆盖 4/10 步全部时刻；BS1/16 共用尺度。32 组未参与校准的观测（episode 48–49）验证，同输入/同噪声对照 FP32；下表取 BS1/16 较差值。

| 模型 | 精度 | 主干特征 cosine | 4步动作相对L2误差 | 10步动作相对L2误差 |
|---|---|---:|---:|---:|
| DINOv3 ConvNeXt-B | FP16 | 0.999806 | 0.055% | 0.049% |
| DINOv3 ConvNeXt-B | INT8 | 0.759517 | 3.026% | 2.946% |
| EfficientViT-B2 native | FP16 | 0.997840 | 0.124% | 0.118% |
| EfficientViT-B2 native | INT8 | 0.217113 | 29.027% | 29.025% |

本次 INT8 PTQ 出现明显数值退化，INT8 延迟不代表等精度加速。

INT8：W8A8，4 个适配器 stride2 卷积保留 FP16，其他非量化算子用 FP16/FP32，后处理 FP32；实际 Conv/GEMM INT8 层数（主干 / 条件编码 / FM 单步）：

| 模型 | BS1 | BS16 |
|---|---:|---:|
| DINOv3 ConvNeXt-B | 76 / 82 / 11 | 76 / 82 / 11 |
| EfficientViT-B2 native | 100 / 106 / 11 | 100 / 106 / 11 |

计时不含 H2D/D2H、视频解码、噪声生成、构建/校准/捕获；未测闭环成功率。PyTorch 仅导出与数值校验，计时进程未加载 PyTorch。

PyTorch：训练/通用推理框架；TensorRT：模型编译优化与低精度推理运行时；CUDA Graph：复用 GPU 执行序列、减少调度开销，可叠加 TensorRT。[NVIDIA 说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)。

原始汇总与逐次计时数据保存在服务器 `docs/100HzS1/data/`，未纳入 Git；复现：`bash tools/quantization/run_trt_latency.sh`。
