# 100HzS1 EfficientViT 接入与首轮实测

观测日期：2026-10-28。编码器适配和时延对比已完成。首轮训练已结束；当前重训使用单帧观测、horizon24。开环误差可核对输入与动作对齐，不能替代闭环任务成功率。

## 选择与结论

本轮选择 **MIT EfficientViT-B2-r256 ImageNet-1K 预训练权重**训练，另提供 B3-r256 配置用于切换。选 B2 的依据是本次 L4 实测的训练批次视觉前向速度和参数量；ImageNet 准确率不能代替机器人任务成功率。

B2 的完整视觉编码在 batch=16 时从 Base 的170.866ms降到81.709ms，即2.09倍速度、耗时降低52.2%；batch=1则为12.784ms，对比Base12.631ms，**当前 eager PyTorch 实现没有单次推理时延收益**。该结果不等于整个训练或整次动作预测加速2.09倍。

权重来自论文作者训练、timm 移植的分类模型，保留空间特征主干，去掉分类头。与 DINOv3 的自监督预训练不同，表示质量差异需闭环测量。来源：[MIT 分类模型说明](https://github.com/mit-han-lab/efficientvit/blob/master/applications/efficientvit_cls/README.md)、[B2 模型卡](https://huggingface.co/timm/efficientvit_b2.r256_in1k)、[B3 模型卡](https://huggingface.co/timm/efficientvit_b3.r256_in1k)。官方 Jetson/A100 TensorRT FP16 表格和这里的 L4 PyTorch BF16 条件不同。

## 维度与分辨率

本节保留首轮 `projected` 视觉时延；最新 native 与完整策略的 GPU 实测、结论见文末“L4完整策略复测”。

| 项目 | DINOv3 ConvNeXt-Base | EfficientViT-B2 | EfficientViT-B3 |
|---|---|---|---|
| 读取图像，CHW | 3×240×320 | 相同 | 相同 |
| 实际输入主干 | 3×256×320 | 相同 | 相同 |
| 视角 / 图像历史 | 左、右、头三视角 / 1帧 | 相同 | 相同 |
| 最终空间下采样 | 32 | 32 | 32 |
| 空间网格 / 每视角tokens | 8×10 / 80 | 相同 | 相同 |
| 原生通道 | 1024 | 384 | 512 |
| 进入融合器前通道对齐 | Identity | 可训练Linear(384,1024) | 可训练Linear(512,1024) |
| 对齐层参数 | 0 | 394240 | 525312 |
| 主干参数（无分类头） | 87566464 | 14977008 | 38997440 |
| 完整视觉模块参数 | 145279104 | 73083888 | 97235392 |
| MultiViewFuser | 宽1024、16heads | 相同 | 相同 |
| ResNetAdapter | conv_num=2，每视角2×2tokens，通道1024 | 相同 | 相同 |
| 交给策略的视觉特征 | [B,12,1024] | 相同 | 相同 |

240→256来自原有VFA `preprocess_images(...,patch_size=32)` 的对齐规则；本次沿用该规则，没有改成224方图或使用分类模型默认裁剪。三种编码器逐像素核对了共同预处理结果；归一化均值[0.485,0.456,0.406]、标准差[0.229,0.224,0.225]，不二次除255。离线数据先按原有 `bgr_to_rgb=true` 纠正，模型输入RGB。

主干默认冻结。EfficientViT的BatchNorm也固定在eval模式，防止外层policy.train()改变预训练统计；新增对齐层、既有融合器/适配器和策略继续训练。保留backbone_trainable开关可进一步解冻。去掉分类头后B2/B3主干约15M/39M，因此不能直接和完整分类器24M/49M参数量混读。

## 实测方法与结果

machine2同一张NVIDIA L4，驱动535.129.03，PyTorch2.7.1+cu126、timm1.0.15、transformers4.56.0。FP32权重+BF16 autocast、eval/inference模式，TF32关闭，不编译。每项30次预热、150次同步计时，顺序测试三种模型。batch表示同时处理的观测数，每份观测含3张图，batch16即48张图。

下表单位ms，使用同步墙钟中位数/p95。完整视觉编码包含共同预处理、主干、通道对齐、多视角融合和适配器；不包含数据读取、解码、BGR转换、CPU→GPU搬运和动作头。显存为该测量进程完整视觉模块及测试张量的peak allocated，不是完整训练显存。

| 模型 | batch | 原生主干 | 主干+对齐 | 完整视觉 | 完整p95 | 显存MiB |
|---|---:|---:|---:|---:|---:|---:|
| dinov3_convnext_base | 1 | 10.012 | 10.013 | 12.631 | 12.735 | 725.2 |
| dinov3_convnext_base | 16 | 157.416 | 158.394 | 170.866 | 172.849 | 1465.8 |
| efficientvit_b2 | 1 | 10.198 | 10.235 | 12.784 | 12.914 | 438.3 |
| efficientvit_b2 | 16 | 69.306 | 69.559 | 81.709 | 82.741 | 987.6 |
| efficientvit_b3 | 1 | 15.247 | 15.307 | 17.853 | 17.971 | 539.2 |
| efficientvit_b3 | 16 | 139.076 | 139.330 | 151.674 | 152.563 | 1215.2 |

测量工具为 `tools/benchmark_visual_backbones.py`，输出 JSON/CSV，包括 CUDA event 和同步墙钟时间、峰值显存、吞吐和形状。原测量共18项，已核对正常退出及完成标记。

## 修改的位置与切换

所有代码直接在开发机 `/data/vepfs/licheng.yang/100HzS1` 修改。

- `src/100hzs1/models/policy/flow_matching/emvis/encoder/efficientvit_encoder.py`：timm MIT B系列包装器、严格加载本地safetensors、空间token格式、坐标和冻结BN。
- `src/100hzs1/models/policy/flow_matching/emvis/embodied_vision.py`：新增efficientvit分支及通道对齐，保留DINO分支初始化、参数键和输出。token_forward仍接受已对齐的1024维特征。
- `configs/100hzs1_train/observation_config/efficientvit_b2_3v.yaml`、`efficientvit_b3_3v.yaml`：完整视觉配置。
- `configs/100hzs1_train/100hzs1_blocks_ranking_rgb_efficientvit_b2_3v_8gpu_L4.yaml`、B3对应文件：独立实验配置，继承相同训练配方。
- `tools/benchmark_visual_backbones.py`、`tools/download_efficientvit_weights.py`、`tools/check_policy_checkpoint.py`：可复现测量、固定版本下载与严格离线策略重载验证。
- `tests/test_visual_backbones.py`：DINO回归、形状、冻结BN、投影梯度、严格加载和离线重建测试。

通过每次实验的顶层YAML选择完整observation配置；例如defaults中使用 `observation_config/efficientvit_b2_3v`，切B3则改为 `observation_config/efficientvit_b3_3v`，切回Base则用 `observation_config/dinov3cnnl_3v_cnn_b`。不要只改一个backbone_target而保留另一模型的参数/权重目录。

启动B2示例（已有正式训练在跑，不要重复执行）：

```bash
cd /data/vepfs/licheng.yang/100HzS1
NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 \
  bash scripts/train/train.sh \
  configs/100hzs1_train/100hzs1_blocks_ranking_rgb_efficientvit_b2_3v_8gpu_L4.yaml
```

Base顶层配置为 `100hzs1_blocks_ranking_rgb_dinov3cnnb_3v_8gpu_L4.yaml`。每次新实验应修改output_dir避免覆盖产物。整个策略checkpoint包含新主干和对齐层，重新加载时load_backbone_pretrain=false，无需再下载原始预训练文件。

## 当前训练配方

四份 blocks_ranking_rgb 顶层配置都设为 `n_obs_steps=1`、`n_image_obs_steps=1`、
`horizon=24`，即当前状态和各视角当前图像，预测未来23帧 joint14 绝对关节动作。
Base 与 B2 使用同样50个 episode、seed42、49训练/1验证、全局batch128、
64个数据worker、200epochs、lr1e-4、BF16和BGR→RGB，视觉主干冻结。

当前输出使用 `MODEL_3v_obs1_h24_8gpu_L4_20261028`，旧训练输出保留。
W&B 离线文件位于相应输出目录的 wandb 子目录。
共享内存帧缓存属于各训练节点，和磁盘 Arrow 缓存、模型权重分别管理。

首轮旧权重（3帧状态/horizon8/未来5动作）在同一验证 episode4 的开环关节
RMSE：Base 0.01566 rad、B2 0.01532 rad。流程见
[开环评测](../scripts/openloop_eval/README.md)。目前暂停100集闭环评测；
没有可用于宣称任务成功率提升的完整结果。自有仿真环境搭建见
[RoboTwin README](../simulation/robotwin2.0/README.md)。


## 原生宽度融合

`EmVisRM` 新增 `efficientvit_fusion_mode`，只影响 EfficientViT：

- `projected`：保持原结构，先 Linear(384/512, dim_2d)，再融合、压缩。
- `native`（代码与共享配置默认）：从 `backbone.out_channels` 自动取得 B2=384、B3=512，忽略旧的 `dim_2d`；
  前置投影为 Identity，MultiViewFuser 和 ResNetAdapter 输入都使用原生宽度。
  `model_adapter_config.dim_out` 仍设为 1024，末尾已有的 Linear 完成升维，
  因而输出仍为 `[B,12,1024]`，后续 scene_adaptor / DiT 接口不变。

保持融合 depth=1、16 heads、MLP ratio=2、conv_num=2：

| 主干 | projected 融合/压缩宽度 | native 融合/压缩宽度 | projected 投影+融合+压缩参数 | native 对应参数 |
|---|---:|---:|---:|---:|
| B2 | 1024 | 384 | 58,106,880 | 8,371,840 |
| B3 | 1024 | 512 | 58,237,952 | 14,701,056 |

表中不含冻结主干和 DiT。宽度可被16整除；此变更未调整注意力头数、空间压缩比例或策略配置。
计算量下降不代表端到端时延按相同比例下降，任务成功率也需要重新评测。

新观察配置为 `observation_config/efficientvit_b2_native_3v.yaml`、B3 对应文件；
它们继承旧配置，只设置 `efficientvit_fusion_mode: native`、`dim_2d: null`。
独立训练配置为：

```text
configs/100hzs1_train/100hzs1_blocks_ranking_rgb_efficientvit_b2_native_3v_8gpu_L4.yaml
configs/100hzs1_train/100hzs1_blocks_ranking_rgb_efficientvit_b3_native_3v_8gpu_L4.yaml
```

这两份配置继承原训练配方，但输出目录名含 `_native_`，不会与既有投影实验共用目录。
当前训练默认 native；本包只加载当前 100hzs1 元数据，不提供旧命名 checkpoint 兼容。

native 是新的网络结构，不能通过删除一个权重键、设置 `strict=False` 或恢复旧优化器状态实现等价转换。
切换时应另开实验，继续加载同一 EfficientViT 预训练主干，重新训练融合/压缩层。
后续策略层虽可单独迁移，但这属于显式的部分初始化，不是无缝续训，本次未自动迁移。

验证覆盖 B2/B3 原生宽度前向、反向、冻结 BN、配置 JSON 重建和严格权重重载。
以下 2026-10-28 数值是改名前实验的记录，不是当前默认值或当前包的兼容性承诺。


本次 native 改动的 CPU 验证结果：16 项视觉/历史元数据测试全部通过；
实际运行目录 `efficientvit_b2_3v_obs1_h24_8gpu_L4_20261028` 下的 `checkpoint-10000`
用新版代码通过完整策略的 strict reload，并得到有限动作输出 `[1,23,14]`。
该阶段仅做 CPU 验证、只读取 checkpoint；后续 GPU 时延实测见下节，未测量任务成功率。


## 2026-10-28 L4完整策略复测：最终结论

当前 PyTorch eager BF16 实现，单机器人 batch=1 下，B2 投影版和 native 版都没有超过 DINOv3-Base；batch16 的完整推理吞吐与训练计算步有实际优势。K/V缓存已默认开启，它是独立于编码器替换的软件优化。

每个batch样本是一组同步观测（三视角RGB和状态），不是一张图。以下数据使用真实验证episode4、当前观测1帧、horizon24/未来23动作、10步去噪。完整推理每组150次、三轮正反顺序；训练计算每组60步。

| 模型 | batch1无缓存/ms | batch1开缓存/ms | batch1开缓存p95/ms | batch16开缓存/ms | batch16训练计算/ms |
|---|---:|---:|---:|---:|---:|
| DINOv3-Base | 77.51 | 70.78 | 74.42 | 266.73 | 338.93 |
| B2 projected | 79.08 | 72.38 | 74.21 | 173.73 | 240.63 |
| B2 native* | 79.25 | 72.57 | 74.47 | 167.56 | 202.30 |

*native融合/压缩层尚未训练，仅测结构时延；其余主干、策略头和归一化沿用B2权重，不能作为成功率证据。

两边都开缓存时，B2 projected 的 batch16 完整推理吞吐为 Base 的 1.54 倍，训练计算为 1.41 倍；训练测量不含数据读取、H2D、EMA和保存。两类主干都按原训练配置显式冻结。

另做450次相邻请求交错验证，全部6种模型顺序均衡轮换，B2相对Base的成对速度比约0.959；两种B2结构的块bootstrap95%区间均低于1，进一步验证单观测没有提速。该区间仅描述本次运行，不外推到其他硬件或后端。

旧方法复测（仅视觉模块）也支持原结论：batch1 Base12.19ms/B2 projected13.98ms，batch16 Base166.77ms/B2 projected80.32ms。旧表分别为12.63/12.78ms与170.87/81.71ms。旧表只测视觉，不应把其2倍吞吐直接当作整套策略或完整训练的加速比例。

缓存前后六组动作检查最大差异为0，每层K/V投影从10次降到1次，总计80→8次。所有正式计时均无其他GPU计算进程，GPU频率/功率未手工修改。

截图中的官方分类模型表使用Jetson/A100和TensorRT FP16；这里是L4的三视角PyTorch策略，两种口径不能直接比较。单机器人更应看batch1响应延迟和p95，多batch推理也可用于多机器人或并行仿真，并不只代表训练速度。

完整方法、分轮数据、CPU输入/H2D、交错验证和官方来源见 [完整报告](100HzS1/l4_latency_review_20261028.md)。

测试脚本：`tools/benchmark_s1_policy.py`。原始结果：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/benchmarks/20261028_l4_latency_review_v1/{legacy_retest,full_v2,paired_b1}`。


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

脚本：`tools/benchmark_single_image_backbones.py`。原始数据：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/benchmarks/20261028_single_image_bs1_v1`。


## 单图时延的运行时归因

单图eager时延并不直接反映GPU算力成本。最新诊断中，Base/B2 eager为6.70/12.63ms；保持同样计算、改用静态CUDA Graph回放后为4.54/2.25ms，输出最大差异均为0。B2在减少逐算子提交和运行时开销后反超Base。此结果不是已上线的完整策略加速，也不是TensorRT实测。详见 [原因分析](100HzS1/bs1_cause_analysis_20261028.md) 和 [报告目录](100HzS1/README.md)。


## 推理 CUDA Graph 默认开关（2026-10-28 后续实现）

后续实现已将视觉主干 Graph 接入 PyTorch `predict_action(..., use_cuda_graph=True)`，
旧 checkpoint 可直接使用，训练 `compute_loss` 保持 eager。`use_kv_cache` 独立且仍默认开启。
此开关目前仅覆盖视觉主干，未捕获融合、压缩和DiT采样。
前面的时延表保持原 eager 测量口径，不能当作新默认执行方式的数字。
完整策略的新对照、输入/动作一致性及限制见 [推理 CUDA Graph 报告](100HzS1/inference_cuda_graph_20261028.md)。
