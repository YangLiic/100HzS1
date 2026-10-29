# 推理专用 CUDA Graph 默认开关

日期：2026-10-28。本次提交包含推理默认开关、兼容性回归、实测报告和复现脚本。

## 使用方式

```python
policy.eval()
result = policy.predict_action(obs)  # 默认 use_cuda_graph=True, use_kv_cache=True
result = policy.predict_action(obs, use_cuda_graph=False)  # eager 对照
```

这是推理调用的运行时参数，不加入训练 YAML、模型配置或 checkpoint。
旧 checkpoint 不需重训或转换；正在训练的任务不需要改变 pipeline、loss、优化器或权重保存格式。
离线脚本 `scripts/openloop_eval/evaluate.py` 同样默认开启，加 `--no-cuda-graph` 可关闭。
安装版本的 PyTorch 推理入口使用新实现；显式加载 checkpoint 内嵌旧源码或运行 ONNX/TensorRT 时，不由此开关控制。

## 捕获范围与边界

本次只捕获视觉主干 `backbone(images)`，保留其原有输出和精度。
预处理、384/512→1024投影（若存在）、view-aware融合、压缩、scene adaptor和DiT采样仍按原方式运行。
K/V缓存独立开启；随机初始噪声在图外按每次请求正常生成。
不能把本开关描述成“整个策略已经被CUDA Graph捕获”。

`compute_loss`、普通编码器调用、训练模式和启用梯度的编码器调用保持eager。
CPU也直接走eager，不构建图。训练仍保留动态执行和常规调试能力；本次未尝试图化反向传播和优化器。

每个视觉编码器最多保留一个图；输入shape、stride、dtype、autocast、device、stream等改变时重新捕获。
训练、模型移动/类型转换、state_dict加载会清空缓存；缓存不作为参数或buffer保存，也不进入deepcopy。
直接替换主干内部子模块/改变其执行逻辑后，应主动调用
`policy.obs_encoder.scene_encoder.clear_cuda_graph_cache()`。
每次返回独立特征，避免下一次replay覆盖之前的结果。

首次调用需要预热和捕获，增加启动时间和显存；稳态计时排除了这部分。
捕获前检查常见host-sync或动态输出算子（例如`.item()`），不支持时记录warning并对当前签名回退eager。
实际CUDA捕获开始后若发生错误，明确报错并要求以 `use_cuda_graph=False` 重启推理，不假装可以安全恢复。

## 一致性与回归验证

- 新增CUDA Graph测试9项，全部在L4上通过：默认开关、CPU/训练/梯度隔离、返回内存所有权、形状和精度切换、缓存失效、准备失败回退、host-sync预检和捕获中错误显式暴露。
- 本提交的独立快照回归27项通过：K/V缓存6、视觉编码11、checkpoint元数据5、离线评测5；加上CUDA Graph的9项，合计36项。
- 实际DINOv3 Base和B2 projected的checkpoint-10000通过strict加载；native使用相同主干/DiT但新的未训练融合/压缩权重，只做结构对照。
- 每个模型分别测试两个真实观测、更换图像、bs1/bs2、首次捕获时固定随机种子的完整动作；特征和动作最大绝对差均为0。
- `return_flow=True` 的完整flow逐步数组也完全相同。模型state_dict键不增不减，未保存或修改任何checkpoint。

真实数据：blocks_ranking_rgb episode4；一个观测batch=1含三张RGB图与本体状态。
输入原图240×320，沿用原有预处理对齐至256×320；horizon24、未来动作23、10步采样。
NVIDIA L4、PyTorch2.7.1+cu126、FP32原始权重+BF16 autocast、TF32关闭。
每组3轮×50次同步计时（150样本），每轮10次预热，顺序正反交替；检查GPU没有额外计算进程。

## 稳态同步墙钟时延

“视觉条件编码”包含预处理、主干、投影（若有）、融合、压缩和scene adaptor；仅主干被捕获。
“完整策略/GPU输入”从GPU观测开始；“完整策略/CPU输入”包含已解码float输入的H2D。
所有项目都不包含视频解码、相机采集、网络通信或机器人执行；不等于完整闭环周期。

| 模型 | 范围 | eager p50/ms | Graph p50/ms | p50耗时降低 | eager p95/ms | Graph p95/ms |
|---|---|---:|---:|---:|---:|---:|
| DINOv3 Base | 视觉条件编码 | 12.53 | 11.87 | 5.2% | 12.67 | 12.02 |
| DINOv3 Base | 完整策略/GPU输入 | 68.44 | 68.09 | 0.5% | 69.93 | 71.47 |
| DINOv3 Base | 完整策略/CPU输入 | 68.96 | 68.98 | -0.0% | 71.97 | 70.88 |
| B2 projected（已训练旧权重） | 视觉条件编码 | 13.44 | 6.45 | 52.0% | 20.31 | 6.60 |
| B2 projected（已训练旧权重） | 完整策略/GPU输入 | 70.93 | 62.95 | 11.2% | 74.85 | 65.83 |
| B2 projected（已训练旧权重） | 完整策略/CPU输入 | 71.03 | 63.34 | 10.8% | 74.20 | 65.50 |
| B2 native（未训练融合/压缩，结构对照） | 视觉条件编码 | 13.22 | 5.63 | 57.4% | 13.81 | 5.75 |
| B2 native（未训练融合/压缩，结构对照） | 完整策略/GPU输入 | 70.65 | 62.56 | 11.4% | 74.54 | 64.73 |
| B2 native（未训练融合/压缩，结构对照） | 完整策略/CPU输入 | 70.98 | 63.23 | 10.9% | 72.77 | 70.98 |

## 首次调用耗时

这些是清空缓存后新捕获并完成一次完整预测的单次样本，不是稳态延迟。

- dinov3_base: 352.76ms。
- efficientvit_b2_projected: 373.16ms。
- efficientvit_b2_native: 390.45ms。

## 结论

对于正在训练的旧B2 projected结构，已有权重可直接使用Graph；本次完整策略GPU输入p50从70.93ms降至62.95ms，耗时降低11.2%。
同轮DINOv3 Base开启Graph后为68.09ms。
这是视觉主干调度优化带来的完整预测收益，不能把此前裸单图主干约2倍优势直接套到完整策略上。
native变窄与Graph是两种独立变化；native未经训练不能据此判断任务成功率。

本轮测试动作输出完全一致，支持在此次已测配置中无数值损失；仍未开展新一轮机器人闭环成功率评测。

## 执行约束的实际验证

刻意在捕获中调用`.item()`时，当前PyTorch2.7.1出现后续随机操作异常，表明捕获失败不能一律catch后继续eager。
因此新增捕获前检查，并区分准备失败与捕获中失败；测试验证预检拦截后正常随机采样仍可使用。
这是保留eager开关、限定捕获范围的具体原因。
参考：[PyTorch CUDA Graph文档](https://docs.pytorch.org/docs/main/notes/cuda.html#cuda-graphs)、
[PyTorch捕获失败后RNG状态问题记录](https://github.com/pytorch/pytorch/issues/171263)。

原始逐次计时及日志：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/benchmarks/20261028_inference_cuda_graph_v3`。
后续默认命令只测 Graph：.venv/bin/python tools/benchmark_inference_cuda_graph.py --output <全新目录>。
上面的历史 eager/Graph 对照需显式 --include-eager 才会复现；是否追加 eager 对照由用户决定。
机器可读汇总：[data/inference-cuda-graph-summary.json](data/inference-cuda-graph-summary.json)。
