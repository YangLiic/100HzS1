# B0 baseline与共享stage4＋RMS残差MLP：FP16分项推理时间

## 结论口径

5.2%是编码器特征相对FP32的RMSE，并非关节动作误差5.2、更不是成功率下降5.2个百分点。此前5%是探针自设检查线，不是行业标准。
同一ONNX在TensorRT FP32下与PyTorch FP32相对误差约0.0023%；PyTorch自身autocast FP16约5.98%；旧TensorRT FP16约5.19%。这表明原导出数学路径基本正确，特征存在FP16敏感性，不能归因为policy代码计算错误。
原EfficientViT LiteMLA将attention显式转float；新构建器保留其敏感attention、归一化和softmax计算为FP32，卷积/Linear等使用TensorRT FP16。保护后的单编码器相对误差约4.59%。这属于FP16部署的混合精度策略，未改网络、权重或训练。
以下四组使用真实已训练权重，分别做同输入/state/噪声的最终动作数值对照后测速；这些对照不等同于TensorRT闭环成功率验证。

## GPU输入下的整批P50（ms）

| policy | 池化 | 保存归一化 | 视觉token | DiT维度 | 完整FM4 | 视觉编码含融合/MLP | B0（新网络含repeat） | baseline原压缩模块 | DiT纯前向×4 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2×2 | minmax | 12 | 2048 | 12.0831 | 0.5768 | 0.4096 | 0.0744 | 11.1243 |
| baseline | 无 | groot | 18 | 2048 | 12.3952 | 0.6672 | 0.4541 | 0.0911 | 11.3253 |
| stage4＋RMS | 2×2 | minmax | 12 | 1024 | 5.0793 | 0.8692 | 0.7694 | 不适用 | 3.7902 |
| stage4＋RMS | 无 | groot | 18 | 1024 | 5.2328 | 0.8749 | 0.7751 | 不适用 | 3.8201 |

baseline原压缩模块计时包括两个原ResBlock、可选2×2池化与Linear128→1024。新网络没有这两个融合后ResBlock，其token压缩已经计入“B0含repeat”；仍保留可选池化与Linear，适配器独立计时在原始JSON中。不要把新网络的repeat再重复算一次。
视觉编码包括raw RGB预处理、B0/共享stage4、三视角融合、空间适配器/池化/Linear、视觉普通或RMS残差MLP，不含KV生成。完整FM4额外包含KV准备、状态/动作MLP、四次DiT、Euler更新和后处理；DiT纯前向×4使用真实四步轨迹对应的已嵌入输入，不含状态MLP、KV生成或Euler。
各子模块单独编译和计时，TensorRT融合边界与完整pipeline不同，因此分项不能直接相加代替完整推理时间。

## 同步wall与I/O口径

| case | 完整FM4 GPU P95 ms | GPU输入完整FM4 wall P50 ms | 含CPU输入H2D/输出D2H的service wall P50 ms | service wall P95 ms |
|---|---:|---:|---:|---:|
| baseline_pool | 12.1487 | 12.0956 | 13.2335 | 13.2820 |
| baseline_nopool | 12.5268 | 12.4081 | 13.7375 | 13.8022 |
| stage4_rms_pool | 5.0868 | 5.0920 | 6.3145 | 6.3804 |
| stage4_rms_nopool | 5.2479 | 5.2456 | 6.6370 | 6.7385 |

service时间包含Python输入校验、RGB/state/既定noise上传与动作下载；不含相机采集、JPEG编码/解码、网络通信或随机噪声生成。GPU输入表排除了H2D/D2H，输入提前驻留GPU。

## 最终动作数值对照

| case | 四个真实输入中最大动作RMSE | 最大动作绝对误差 | 原框架基础检查线 |
|---|---:|---:|---|
| baseline_pool | 0.0028101 | 0.0156910 | RMSE≤0.01且max≤0.05，通过 |
| baseline_nopool | 0.0023728 | 0.0104201 | RMSE≤0.01且max≤0.05，通过 |
| stage4_rms_pool | 0.0047446 | 0.0305929 | RMSE≤0.01且max≤0.05，通过 |
| stage4_rms_nopool | 0.0055835 | 0.0305411 | RMSE≤0.01且max≤0.05，通过 |

该基础检查线用于捕获明显导出偏差，并非用户任务的成功率验收线；未进行新的RoboTwin闭环评测。所有中间特征和最终动作误差均保留在parity.json中。

## 测试条件与来源

统一三视角、bs=1、标准FM、tau_token、1000时间桶、4次[0,250,500,750]前向、KV开启、CUDA Graph关闭、TensorRT FP16主计算与FP32敏感计算/后处理。raw RGB240×320进入预处理后256×320。
四组均在同一个L4/GPU UUID上计时：`NVIDIA L4, GPU-7766bb82-f822-0968-8b8e-113cb8e06e27, 535.129.03, 2025, 52`；TensorRT 10.9.0.34。build热队列使用其他GPU，避免与测速卡争用；每项预热100次、每轮300次、共3轮。计时进程未导入PyTorch。
池化组采用以前已训练的minmax权重，不池化组采用已训练的groot权重；同一池化设置下baseline/新网络归一化一致。跨池化比较同时存在权重和normalizer差异，不能称为固定同一权重只改池化开关的因果实验。本任务只比较实际网络推理成本，未重训。
精度参考：[NVIDIA Accuracy Considerations](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/accuracy-considerations.html)。

## 产物路径

全部引擎、ONNX、refs、构建详情、数值对照、原始计时：`/data/vepfs/licheng.yang/100HzS1/tools/quantization/outputs/b0_policy_fp16_components_20261102_v1`
封存源码、request、队列、日志与审计：`/data/vepfs/licheng.yang/100HzS1/runs/b0_policy_fp16_components_20261102_v1`

baseline_pool：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_four_variants_fm4_20261102_v1/baseline`

baseline_nopool：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_groot_nopool_baseline_fm4_20261102_v1/baseline`

stage4_rms_pool：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_four_variants_fm4_20261102_v1/stage4_rms`

stage4_rms_nopool：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/stage4_rms`

源码基于commit `7e1dd70ac8542c0c8926c3412cb739581735c9d1`并按source_hashes封存新增量化工具；未push。

## 正式入口与可用缓存

同一精度规则已接入正式 `src/100hzs1/inference/export_tensorrt.py`，并由 `src/100hzs1/inference/fp16_precision.py` 统一管理；`tools/quantization/trt16_build.py` 的backbone入口也使用该规则。policy网络、训练和权重均未修改。旧缓存会因源码哈希变化而失配，避免误用旧精度策略。

生产接回验证：导出模型的Preprocess/Encoder/Step/Post等数学函数AST与已测快照完全相同；24个FP16组件的精度约束列表逐项匹配已测构建记录；19项推理/归一化回归检查通过。28个计划全部核对实际层格式：所有非post计划确有Half计算，INT8格式为0。

四组已发布到原生cache，并用标准load_inference_policy入口逐一加载测试，无猴补ensure_engines、未重新构建、计时/缓存热加载未导入PyTorch；每次真实FM调用4次。可直接选用推理YAML：`/data/vepfs/licheng.yang/100HzS1/tools/quantization/outputs/b0_policy_fp16_components_20261102_v1/deployment_fp16.yaml`。cache根：`/data/vepfs/licheng.yang/100HzS1/tools/quantization/outputs/b0_policy_fp16_components_20261102_v1/native_cache`。

此YAML只选择TensorRT FP16/4-step/KV-on/Graph-off和以上cache路径，state/action的minmax/groot仍遵循各自checkpoint保存语义；它们不是INT8校准方式。生产保护与量化工具代码与本报告一并纳入当前代码提交；测试所用源码、引擎哈希与生产验证证据保持原始封存记录。

生产验证：`/data/vepfs/licheng.yang/100HzS1/runs/b0_policy_fp16_components_20261102_v1/production_cache_verified.json`。原request成功，machine1已恢复holding。
