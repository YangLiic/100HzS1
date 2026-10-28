# L4 离线时延复测：DINOv3-Base 与 EfficientViT-B2

测量日期：2026-10-28。

**本次结论：B2投影版没有带来单观测完整策略提速，Base为70.78ms、B2为72.38ms；batch16完整策略吞吐为Base的1.54倍，batch16训练计算速度为1.41倍。额外450次交错请求验证同一单观测结论。**

**结论仅针对本次 L4、PyTorch eager、BF16、三视角 VFA 实现。** 单机器人主要判断 batch=1 的完整动作预测时延；批量视觉前向更快，不等于单次策略预测更快，也不能直接当作训练加速比例。

## 默认开关与 batch 定义

`predict_action` 的 `use_kv_cache` 已默认 `True`；`model.eval()` 后直接调用即可。False 仅用于对照。每次预测重新编码图像、创建各层 K/V，缓存不会跨预测请求复用。

本报告 batch=1 是一组同步观测：左、右、头三个视角各一张 RGB 图，以及当前14维状态。主干实际处理3张图；batch=16 是16组观测、48张图。历史图像1帧、状态1帧，horizon24、未来动作23帧，Euler去噪10步。

## 1. 与旧文档相同方法的复测

以下只测完整视觉模块，包含预处理、主干、融合和压缩，不包含 DiT、数据解码或 H2D。沿用旧脚本，30次预热、150次同步计时；GPU驻留随机RGB输入，原投影模式。

| 模型 | batch | 旧文档p50/ms | 本次p50/ms | 本次p95/ms | 相对旧值 |
|---|---:|---:|---:|---:|---:|
| dinov3_convnext_base | 1 | 12.63 | 12.19 | 12.34 | -3.5% |
| dinov3_convnext_base | 16 | 170.87 | 166.77 | 168.55 | -2.4% |
| efficientvit_b2 | 1 | 12.78 | 13.98 | 18.54 | +9.4% |
| efficientvit_b2 | 16 | 81.71 | 80.32 | 81.07 | -1.7% |

这组复测用于对照旧文档；下面的真实观测基准使用训练好的 checkpoint、三种结构共同驻留及多轮正反顺序测量，绝对数值应在同一表内比较，不混合不同测试口径。

## 2. batch=1：完整动作预测

GPU驻留真实观测，包含完整视觉编码、scene/state适配器、10步DiT、动作反归一化。每组150样本，分3轮、每轮50样本及20次预热；第二轮反向顺序，第三轮轮换起点。下表均为同步墙钟耗时。

| 模型 | 无缓存p50/ms | 开缓存p50/ms | 开缓存p95/ms | 缓存降低耗时 | 开缓存各轮p50/ms |
|---|---:|---:|---:|---:|---|
| DINOv3-Base | 77.51 | 70.78 | 74.42 | 8.7% | 70.83, 72.47, 70.10 |
| EfficientViT-B2 projected | 79.08 | 72.38 | 74.21 | 8.5% | 71.55, 72.32, 73.04 |
| EfficientViT-B2 native* | 79.25 | 72.57 | 74.47 | 8.4% | 71.17, 72.71, 73.13 |

同样开启缓存时，相对 Base：

- EfficientViT-B2 projected：0.978×速度，耗时变化 +2.3%。
- EfficientViT-B2 native*：0.975×速度，耗时变化 +2.5%。

**native* 是结构时延基准：沿用训练好的B2主干和策略头，384维融合/压缩层重新初始化，尚未训练；不能据此宣称任务成功率。** 两个正式旧结构使用对应 `checkpoint-10000` 严格加载。

附：batch=1完整视觉编码：

| 模型 | p50/ms | p95/ms |
|---|---:|---:|
| DINOv3-Base | 12.45 | 12.63 |
| EfficientViT-B2 projected | 13.49 | 14.70 |
| EfficientViT-B2 native* | 13.60 | 14.07 |

### 2.1 相邻请求交错验证

为排除模型测试顺序和缓慢漂移，另测150组相邻请求，每组依次运行三个模型，各模型顺序按全部6种排列均衡轮换，共450次完整动作预测，均开启缓存。统计速度比为同一组Base耗时/B2耗时；小于1表示B2更慢。每6组作为一个连续块，进行10000次块bootstrap。区间描述此测量序列，不代表跨硬件/部署后端的普遍区间。

| 模型 | p50/ms | p95/ms | 相邻请求速度比/Base | 速度比95%区间 |
|---|---:|---:|---:|---|
| DINOv3-Base | 73.87 | 87.30 | 1.0000 | — |
| EfficientViT-B2 projected | 76.90 | 89.37 | 0.9588 | 0.9566 – 0.9607 |
| EfficientViT-B2 native* | 77.41 | 90.35 | 0.9589 | 0.9558 – 0.9615 |

B2两种结构的速度比区间均低于1，这轮控制顺序后的测试也不支持单观测加速；与三轮主测的定性结论一致。不同运行组之间绝对时延会变动，不能跨组直接相减归因。

## 3. batch=16：并行推理吞吐

| 模型 | 完整视觉p50/ms | 完整策略无缓存p50/ms | 完整策略开缓存p50/ms | 开缓存p95/ms | 开缓存观测组/秒 |
|---|---:|---:|---:|---:|---:|
| DINOv3-Base | 168.93 | 277.91 | 266.73 | 270.55 | 59.99 |
| EfficientViT-B2 projected | 80.56 | 186.15 | 173.73 | 177.31 | 92.10 |
| EfficientViT-B2 native* | 74.82 | 179.95 | 167.56 | 169.51 | 95.49 |

batch16的整批延迟不能除以16后当作单机器人响应时间。它适用于多机器人、多路环境、离线处理等吞吐场景。训练提速需看下面的反向传播和优化器测量。

## 4. 实际训练计算步

每组60次计时，分3轮；包含 compute_loss、backward、梯度裁剪和 AdamW 更新。两类主干均按原训练配置显式冻结。使用 lr=0 保持权重值不变，但仍执行梯度、动量与更新算子；不保存任何权重。不包含数据加载、H2D、EMA与checkpoint保存，因此不是完整训练任务墙钟时间。

| 模型 | batch1计算步p50/ms | batch16计算步p50/ms | batch16相对Base速度 | batch16样本/秒 |
|---|---:|---:|---:|---:|
| DINOv3-Base | 145.53 | 338.93 | 1.00× | 47.21 |
| EfficientViT-B2 projected | 147.71 | 240.63 | 1.41× | 66.49 |
| EfficientViT-B2 native* | 127.96 | 202.30 | 1.68× | 79.09 |

## 5. CPU输入、正确性与环境

单观测从CPU pinned float RGB/状态输入开始，包含H2D及完整策略、开启缓存：

| 模型 | p50/ms | p95/ms |
|---|---:|---:|
| DINOv3-Base | 70.38 | 77.98 |
| EfficientViT-B2 projected | 73.26 | 83.44 |
| EfficientViT-B2 native* | 72.23 | 74.77 |

CPU输入与GPU输入是独立计时组，小幅非单调差别来自运行波动，不能直接相减估计H2D成本。

三种模型、两种batch的缓存正确性检查全部通过；最大动作差异 0。每层K/V调用从10次变成1次，全模型从80次变成8次。

真实输入来自 blocks_ranking_rgb 验证episode4的16个时间点，离线BGR只转RGB一次。240×320经共同预处理变为256×320，每视角80个空间token。所有模型使用同一组输入；视频解码、相机采集和网络传输不在计时内。

设备：`NVIDIA L4, GPU-1ec94d36-e6ba-3cdf-4339-81b1950a9818, 535.129.03, 72.00 W, 2040 MHz`。Torch 2.7.1+cu126、CUDA 12.6、timm 1.0.15、Transformers 4.56.0。CPU单线程；TF32关闭；不编译；原checkpoint参数dtype、BF16 autocast。记录CUDA Event和同步墙钟，正式结果使用后者。

GPU开始时无其他计算进程，各测量段前检查进程隔离；不改变GPU频率/功率设置，不中断已有训练。原生尺寸模式无新训练checkpoint。

训练公平性：DINO历史加载器在不重载预训练时不会恢复冻结标志，而requires_grad不保存在checkpoint中；本测试在三种模型上均显式恢复backbone_trainable=False。未修正的先导训练数据、已中断的full/数据均未纳入本报告。

## 6. 与截图中官方表格的关系

截图是分类模型比较。官方说明使用Jetson Nano、AGX Orin和A100，TensorRT FP16且包含数据传输；256方图、分类头和比较对象也与三视角DINOv3-VFA不同。因此不能把截图中的2.8ms或A100 images/s直接当作这里的机器人策略时延。来源：[MIT EfficientViT分类模型说明](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_cls/README.md)。

本报告不测机器人闭环成功率，不比较TensorRT，不把参数/MAC减少等同于时延按比例下降。缓存收益应在同一编码器内比较；换编码器收益应在两边都开启缓存的条件下比较。

## 原始记录与复现

远端结果：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/VFA/benchmarks/20261028_l4_latency_review_v1`。最终正式数据只使用 `legacy_retest/`、`full_v2/` 和 `paired_b1/`。

测试脚本：`/data/vepfs/licheng.yang/VFA/tools/benchmark_vfa_policy.py`；源文件SHA256、checkpoint配置SHA256、输入索引、逐次耗时、各轮统计、GPU温度/频率/功率状态均保留。

原始数据包包含旧方法JSON/CSV、正式基准JSON/CSV/JSONL、正确性检查、环境和测试脚本快照。


## 补充：真正单张图 batch=1 的编码器测量

此前“观测batch=1”包含3张图。本节输入严格为 `[1,3,H,W]`，只比较裸空间特征主干，不含预处理/H2D、分类头、外部Linear、融合、压缩和DiT。DINO取stages输出，B2取forward_features输出，均保留各自主干内部操作。

| 精度 | H×W | DINOv3-Base p50/ms | B2 p50/ms |
|---|---|---:|---:|
| BF16 autocast | 256×320 | 6.663 | 10.098 |
| BF16 autocast | 256×256 | 6.869 | 10.150 |
| FP16 autocast | 256×320 | 6.494 | 9.504 |
| FP16 autocast | 256×256 | 6.534 | 9.590 |

结果：本次L4/PyTorch eager环境中，B2在上述四组单张图条件下都没有更快，耗时约高46%–52%。保留VFA实际编码器的最终归一化和token/位置包装时也得到相同结论。

每组450样本（3轮×150），每段30次预热。FP32权重，TF32关闭，不编译；FP16 autocast也不等同于TensorRT FP16。尚不能外推到其他部署后端。

脚本：`tools/benchmark_single_image_backbones.py`。原始数据：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/VFA/benchmarks/20261028_single_image_bs1_v1`。
