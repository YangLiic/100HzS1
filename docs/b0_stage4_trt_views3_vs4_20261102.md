# B0共享stage4：TensorRT三视图与四视图延迟

整批GPU P50：三视图0.752192ms，四视图0.788272ms，增加4.797%（0.036080ms）。并非严格相等；两者保留一位小数都会显示0.8ms。

## 输入与等价关系

当前EfficientViTEncoder将[B,V,C,H,W]展平为[B×V,C,H,W]再编码，所以B=1时V=3/4就是有效图像batch=3/4。已用同一权重逐值验证五维与对应四维输入的FP32输出完全相等。三视图和四视图的前3张预处理图像相同，第四张也取自已有真实相机数据。

| 输入(B,V,C,H,W) | 内部图像batch | GPU P50 ms | GPU P95 ms | 同步wall P50 ms | 同步wall P95 ms |
|---|---:|---:|---:|---:|---:|
| (1,3,3,256,320) | 3 | 0.752192 | 0.872474 | 0.761955 | 0.889322 |
| (1,4,3,256,320) | 4 | 0.788272 | 0.804325 | 0.798247 | 0.817606 |

## 测试口径

GPU：NVIDIA L4, GPU-1ec94d36-e6ba-3cdf-4339-81b1950a9818, 535.129.03, 72.00；TensorRT 10.9.0.34；CUDA runtime 12.8；Torch 2.7.1+cu126仅在独立CPU导出进程使用，TensorRT计时进程未导入Torch。
使用已完成55/100组的main权重，仅测EfficientViT-B0原骨干＋同一个stage4额外复用两次＋共享128→64桥接，输出每视图6×128。融合器、视觉侧MLP、DiT以及图像预处理、H2D/D2H不在本表计时范围内；本表不是完整policy延迟。
FP16 enabled、FP32 I/O、TF32关闭、CUDA Graph关闭。分别为V=3/4设置min=opt=max静态profile，来自同一ONNX和同一权重，并非同一动态引擎。两个引擎优化后分别196/194层，格式包含Float/Half，不能称为所有算子纯FP16或INT8测试。原有native builder的profiling_verbosity=DETAILED保持一致。
每个case每轮预热200次、测量500次，共6轮，每组3000个样本；AB/BA交替次序。计时前已分配buffer并将输入放在GPU。用CUDA事件记录单次整批耗时，同时用perf_counter_ns记录含同步的wall时间。保留全部样本，包括V=3第三轮的抖动，未剔除异常轮次。

## 数值检查：尚未达到初设检查线

| V | 输出shape | 相对FP32 RMSE | 余弦相似度 | 有限值 | 初设5%相对RMSE检查线 |
|---:|---|---:|---:|---|---|
| 3 | [1, 3, 6, 128] | 5.1913% | 0.998652294 | 是 | 未通过 |
| 4 | [1, 4, 6, 128] | 5.2815% | 0.998605688 | 是 | 未通过 |

初次bench因V=3的相对RMSE约5.19%超过初设5%检查线而停止，尚未产生时延样本。保留原失败日志后，第二次将数值通过与否改为显式记录，使用原有两个引擎完成延迟测试；没有重建引擎、改精度或删除不利样本。两组都未通过初设检查线，因此本结果只用于batch时延比较，不能据此认定量化精度已达标。后续正式量化需处理数值误差并验证策略成功率。

## 解释与限制

NVIDIA说明batch能够并行处理多个输入、摊薄每层启动与同步开销，并影响硬件利用率。因此整批延迟不必随图像数线性增长，与本次图像数增加33.3%而P50仅增加约4.8%的结果一致。这是基于文档和测量的解释；未做逐kernel归因，也未证实TensorRT会将batch=3补齐成4。不同GPU、尺寸、精度和优化profile需要重新测量。
[NVIDIA TensorRT batching](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html#batching)，[NVIDIA CUDA事件计时](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/measurement-techniques.html#cuda-events)。

## 路径与版本

此前清理改动commit：`7e1dd70ac8542c0c8926c3412cb739581735c9d1`，未push。本次新增测速脚本和本报告尚未commit。
脚本：`/data/vepfs/licheng.yang/100HzS1/tools/quantization/benchmark_trt_view_batch.py`
原生运行时：`/data/vepfs/licheng.yang/100HzS1/tools/quantization/benchmark_trt_latency.py`
ONNX、引擎、构建详情、原始测量及汇总：`/data/vepfs/licheng.yang/100HzS1/tools/quantization/outputs/b0_stage4_views3_vs4_20261102_v1`
构建及初次检查日志：`/data/vepfs/licheng.yang/100HzS1/runs/trt_view_batch_20261102_v1/runner.log`
最终测速日志：`/data/vepfs/licheng.yang/100HzS1/runs/trt_view_batch_20261102_v1/bench_retry.log`
checkpoint：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/stage4_rms`
main SHA256：`204949a0096ef4d6bd428babdbefc5ab7d8d738b8a3ee00116e8e1b8e6ab9e66`
ONNX SHA256：`7660fcb5d351f7d60d965957efba7acdfda347f7079cd3a5a6566b74983e86e8`
测速脚本SHA256：`177a53925d91b685d255c21e6da9eecc5ceaf60019c9d90e965e930f9cc6e086`
