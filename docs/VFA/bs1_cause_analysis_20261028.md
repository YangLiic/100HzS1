# 为什么 EfficientViT-B2 的 bs1 没有提速

日期：2026-10-28。范围：同一张NVIDIA L4，输入 `[1,3,256,320]`，裸空间主干；FP32权重、BF16 autocast、TF32关闭、eval/inference。没有384→1024投影、VFA融合、压缩或DiT。

## 结论

**当前PyTorch eager路径中，B2主要受到执行调度、CPU逐算子发射与碎片化算子路径的限制；较低的GPU计算工作量没有转化成较低的单次墙钟延迟。** 这不意味着B2在所有后端都比Base慢。静态CUDA Graph对照中，B2已反超Base，但这仍是裸骨干诊断，不是已交付的完整VFA加速。

## 直接证据

| 模型 | eager中位数/ms | CUDA Graph回放中位数/ms | profiler GPU事件/次 | profiler GPU事件时长合计/ms |
|---|---:|---:|---:|---:|
| DINOv3-Base | 6.70 | 4.54 | 641 | 4.57 |
| EfficientViT-B2 | 12.63 | 2.25 | 865 | 3.25 |

每个时延项30次预热、200次同步计时；Graph保持相同FP32权重、相同BF16 autocast运算与相同输入，并未将模型改成另一种精度。两个模型Graph与eager的空间特征输出最大绝对差均为0。

profiler对每个模型另外采样5次前向。其GPU事件时长含仪器开销，不能与未profiling的墙钟时延直接相减、推算精确CPU占比，也不能把重叠的CPU/GPU算子聚合项相加。表中的事件总时长只来自CUDA设备事件。Graph的结果属于另一个运行方式，不能直接替换此前eager报告的数字。

Graph重用同一个静态执行图，减少逐算子CPU提交、运行时与内存分配路径，并改善GPU任务衔接；它不是TensorRT，也不是已进行跨层算子融合的证明。此对照强烈支持运行时开销是当前B2单图瓶颈的重要部分。

## 源码与算子证据

### 1. 参数少不等于发射的算子少

当前主干模块清单：

- DINOv3-Base：40个Conv2d、72个Linear；ConvNeXt stage中36个重复层。
- B2：100个Conv2d、46个BatchNorm2d、10个LiteMLA，存在多段Conv/Norm/激活、depthwise卷积及多尺度注意力路径。

这是模块数，不是FLOPs或延迟。profiler补充了实际执行证据：B2每次约865个GPU事件，Base约641个；B2有更多小算子和卷积运行时调用。B2的GPU事件时长合计反而较少。不能仅凭模块数解释全部时间差，但它与Graph的因果对照方向一致。

### 2. 当前B2的轻量线性注意力并非全程低精度

安装的timm 1.0.15中，`LiteMLA._attn` 会执行 `q.float(), k.float(), v.float()`，并在关闭autocast的上下文中进行两次矩阵乘法、归一化除法，最后转回原dtype。因此，即使外层打开BF16/FP16 autocast，也不能把整段注意力当作全低精度融合实现。

源码：`/data/vepfs/licheng.yang/VFA/.venv/lib/python3.10/site-packages/timm/models/efficientvit_mit.py` 的 `ConvNormAct.forward`、`LiteMLA._attn` 与 `LiteMLA.forward`。这是当前安装实现的事实，不表示所有EfficientViT实现都相同，也不表示应直接去掉FP32稳定性处理。

### 3. dtype转换也有开销，但不是B2比Base慢的充分解释

原始autocast路径的 `aten::_to_copy` 调用/次：Base334、B2165。双方都有转换，且Base更多；不能把B2慢单独归因于转换。

另做纯BF16权重/输入诊断，Base/B2 eager中位数为5.40/11.25ms，B2仍未反超。直接转换权重会改变数值；此项不是已验证等价的部署方案，也未修改或保存训练checkpoint。主要结论来自数值一致的Graph对照，而非这个dtype消融。

## 为什么多batch有收益

batch增大时，每次发射所处理的数据更多，固定的调用和调度成本被更多图片摊销，GPU计算占比增加。此前图像batch48（观测batch16）中，B2的编码和完整策略吞吐已显示优势。这与单图eager不占优可以同时成立。

## 与论文结果的关系

官方分类表使用TensorRT FP16，在Jetson/A100上测量。其优化执行路径不同于这里的PyTorch逐算子eager。当前Graph实验说明，改变执行方式后B2的排序确实可以反转，但不能把Graph数字等同于论文TensorRT数字，也不能保证完整VFA或闭环成功率按相同比例改善。

参考：[EfficientViT官方分类说明](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_cls/README.md)、[NVIDIA TensorRT性能指南](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)。本报告的归因主要来自本地测量与源码，不是从论文时延外推。

## 后续优先级

1. 保持K/V缓存默认开启；已验证它不改变动作输出。
2. 单独开发静态shape视觉CUDA Graph或TensorRT路径，再测三视角、包含数据搬运与10步DiT的完整预测。
3. 不把B2 native变窄自动等同于低延迟；它主要减少融合/压缩容量，闭环任务性能仍需训练和验证。

本次只做诊断并保存报告，没有启用Graph生产路径、修改模型权重或改变已有训练。原始trace位于 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/VFA/benchmarks/20261028_bs1_cause_profile_v1`，汇总保留GPU事件、算子次数、各项中位数和输出差异。
