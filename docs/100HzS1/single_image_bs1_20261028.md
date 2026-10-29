# L4 真正单张图 bs=1：DINOv3-Base vs EfficientViT-B2

日期：2026-10-28。输入严格为 `[1,3,H,W]`，不是三视角观测。

**结果：在本次L4、PyTorch eager、FP32权重加BF16/FP16 autocast的四组条件中，B2裸空间特征主干均未超过DINOv3-Base，中位耗时约高46%–52%。该结论仅适用于本次实现和运行条件。**

## 裸空间特征主干

DINOv3直接运行所有ConvNeXt stages，输出stride32特征图；EfficientViT直接运行 `forward_features`。不计DINOv3末尾全局汇聚和token归一化、坐标/token包装、预处理、H2D、分类头、外部Linear、多视角融合、压缩或DiT。EfficientViT自身的Attention属于主干，正常保留。

| 精度 | 输入H×W | Base p50/ms | Base p95/ms | B2 p50/ms | B2 p95/ms | B2耗时变化 |
|---|---|---:|---:|---:|---:|---:|
| BF16 autocast | 256×320 | 6.663 | 6.967 | 10.098 | 10.654 | +51.6% |
| BF16 autocast | 256×256 | 6.869 | 7.136 | 10.150 | 10.672 | +47.8% |
| FP16 autocast | 256×320 | 6.494 | 6.652 | 9.504 | 9.789 | +46.4% |
| FP16 autocast | 256×256 | 6.534 | 6.692 | 9.590 | 9.843 | +46.8% |

## 实际VFA单视角编码器接口

保留DINOv3最终归一化以及两种编码器的token/位置包装，但不接任何VFA后续模块。用于确认裸主干比较未因去掉输出包装而改变结论。

| 精度 | 输入H×W | Base p50/ms | B2 p50/ms |
|---|---|---:|---:|
| BF16 autocast | 256×320 | 6.869 | 10.289 |
| BF16 autocast | 256×256 | 6.864 | 10.375 |
| FP16 autocast | 256×320 | 6.563 | 9.801 |
| FP16 autocast | 256×256 | 6.656 | 9.678 |

## 方法与范围

- 每组3轮、每轮30次预热和150次计时，共450样本；反向顺序及轮换起点。
- 同一张空闲L4，两种模型均eval/inference模式，CPU单线程；TF32关闭，不编译，不改频率和功率上限。
- FP32权重，分别使用BF16和FP16 autocast；FP16这一行不是TensorRT部署结果。
- GPU驻留的相同RGB随机输入，ImageNet归一化在计时外完成。
- 同时保留CUDA Event与同步墙钟，表格使用同步墙钟；逐轮p50及全部逐次计时保留在JSON。
- 两个模型都加载既有本地预训练权重；B2不含分类头，没有384→1024投影。
- 测量前及各段前后检查GPU计算进程隔离。测试正常退出，不保存或改变任何训练权重。

这些数值是图像batch=1，不能与上一轮图像batch=3的观测batch=1混用；也不包含动作预测，因此不能直接作为机械臂完整控制延迟。

## 复现与原始记录

远端脚本：`/data/vepfs/licheng.yang/100HzS1/tools/benchmark_single_image_backbones.py`。
远端结果：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/benchmarks/20261028_single_image_bs1_v1`。

原始包包含脚本、环境/权重路径/脚本SHA256、16组汇总、48段逐次计时和完成标记。
