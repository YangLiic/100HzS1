# EfficientViT B0/B1 编码延迟（2026-10-29）

选择同一 MIT EfficientViT B 系列；B0/B1 均有 ImageNet-1K 预训练权重。去除分类头，保持原生通道，不添加投影。

| 主干 | 预训练版本 | 含分类头参数/M | 实测主干参数/M | tokens 通道 |
|---|---|---:|---:|---:|
| B0 | r224_in1k | 3.41 | 0.683 | 128 |
| B1 | r256_in1k | 9.10 | 4.644 | 256 |
| B2 | r256_in1k | 24.33 | 14.977 | 384 |

来源：[B0](https://huggingface.co/timm/efficientvit_b0.r224_in1k)、[B1](https://huggingface.co/timm/efficientvit_b1.r256_in1k)、[MIT 官方模型](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_cls/README.md)。参数量由本地 timm 1.0.15 统计。

NVIDIA L4（72W，默认动态时钟），TensorRT 10.9.0.34，TF32/CG 关闭。BS1=3视角、BS16=48张RGB；所有模型统一每视角3×256×320，输出每视角80个原生tokens。仅计主干，排除预处理、H2D/D2H、融合及策略。每项3轮，每轮预热30次后同步计时100次；下表为整批 p50/p95，ms，不除以BS。

| 主干 | 精度 | BS1 p50 / p95 | BS16 p50 / p95 |
|---|---|---:|---:|
| EfficientViT-B0 | FP16 | 0.393 / 0.657 | 4.026 / 4.094 |
| EfficientViT-B0 | INT8 | 0.716 / 0.777 | 7.097 / 7.381 |
| EfficientViT-B1 | FP16 | 0.787 / 0.809 | 12.582 / 13.148 |
| EfficientViT-B1 | INT8 | 1.523 / 1.543 | 22.499 / 23.128 |
| EfficientViT-B2 | FP16 | 1.629 / 1.660 | 29.489 / 30.743 |
| EfficientViT-B2 | INT8 | 2.964 / 2.990 | 51.426 / 52.710 |
| DINOv3 ConvNeXt-B | FP16 | 5.861 / 5.925 | 114.501 / 117.879 |
| DINOv3 ConvNeXt-B | INT8 | 7.807 / 7.958 | 124.882 / 126.971 |

B0 FP16 BS1 三轮 p50 为 0.392/0.623/0.393 ms，存在轮间波动；表中保留全部样本。

| 主干 | 精度 | 特征 cosine（较差BS） | 特征相对L2（较差BS） |
|---|---|---:|---:|
| EfficientViT-B0 | FP16 | 0.998378 | 5.71% |
| EfficientViT-B0 | INT8 | 0.155060 | 984.37% |
| EfficientViT-B1 | FP16 | 0.995382 | 9.60% |
| EfficientViT-B1 | INT8 | 0.074245 | 874.82% |
| EfficientViT-B2 | FP16 | 0.997840 | 6.59% |
| EfficientViT-B2 | INT8 | 0.217113 | 888.03% |
| DINOv3 ConvNeXt-B | FP16 | 0.999806 | 1.97% |
| DINOv3 ConvNeXt-B | INT8 | 0.759517 | 68.53% |

当前 INT8 配方对 B0/B1 未获得速度收益，且特征明显失真，不作为可用部署结果。FP16 也有表中数值偏差，不能据此保证闭环精度。

128组真实观测校准（episode0–7），32组独立观测校验（episode48–49），对照各模型FP32特征。INT8沿用Entropy2显式Q/DQ混合精度配方；未量化算子保留FP16/FP32。PyTorch仅导出与FP32校验，计时进程不加载PyTorch。B2/DINO使用原实验引擎重新测量。

这轮不含闭环成功率；主干缩小不等于任务精度保持。原始数据：`docs/100HzS1/data/efficientvit_small_20261029_summary.json` 和同名前缀 measurements.jsonl。复现：`.venv/bin/python tools/quantization/benchmark_efficientvit_small.py --root work/efficientvit_small_20261029`。

FP16 相对 B2 的 p50 加速：EfficientViT-B0 BS1 4.14× / BS16 7.32×；EfficientViT-B1 BS1 2.07× / BS16 2.34×。
