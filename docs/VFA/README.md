# VFA 实验与推理优化报告

- [L4完整策略时延复测](l4_latency_review_20261028.md)：三视角、K/V缓存、B2 projected/native、batch1/16与训练计算。
- [真正单张图bs1编码器测试](single_image_bs1_20261028.md)：裸主干与实际编码接口，BF16/FP16、矩形/方形输入。
- [单张图未提速的原因分析](bs1_cause_analysis_20261028.md)：profiler算子与GPU事件、CUDA Graph因果对照、精度转换消融。

- [推理 CUDA Graph 默认开关](inference_cuda_graph_20261028.md)：运行时参数、训练隔离、旧权重一致性与完整策略时延。

图表位于 `figures/`，机器可读汇总位于 `data/`。原始逐次计时和trace路径见各报告。

注意：native融合/压缩层尚未训练，只报告结构时延；早期CUDA Graph报告是裸主干诊断对照；最新PyTorch推理已默认启用视觉主干CUDA Graph和K/V缓存，范围与验证见新增报告。
