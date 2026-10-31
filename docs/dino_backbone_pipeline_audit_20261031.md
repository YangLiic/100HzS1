# DINO Base 与 EfficientViT 的训练、推理及闭环排查

日期：2026-10-31。对象：beat_block_hammer 已完成的 16 组与 handover_block 当前已完成权重。正式 handover 训练和闭环继续运行；本次测试独立使用 local machine 的 NVIDIA L4，不改变正式权重、运行脚本、冻结快照或原结果。

状态：代码、权重、数值审计、同100种子20-step诊断及两个成功种子的同机4-step对照均已完成并独立核验（2026-10-31 10:50 CST）。本轮排查完成；DINO整体成功率差距的根因仍未得到因果确认。

## 当前判断

**目前没有定位到 DINO 专属的漏载预训练、颜色翻转、缓存错帧、token 丢失、视觉梯度断开或 KV 缓存数值错误。DINO 的闭环差距确实可复现，但现有证据不足以把原因归结为“DINO 特征更差”，也不足以断言存在某个已查明的软件 bug。**

已经确认两项会影响解释的重要事实：

1. 当前所谓 DINOv3 Base 是 **DINOv3 ConvNeXt-Base**，不是 DINOv3 ViT-Base。DRRM 的部分历史配置使用 ConvNeXt-Large，另一些使用 ViT-Base，不能仅按“都叫 DINO”视为同一个 backbone。
2. 当前比较同时改变了视觉 backbone 和随机初始化的视觉融合、压缩头：DINO 这部分约 57.71M 可训练参数，B0 约 1.02M，约差 56.5 倍。该倍数仅指这两个视觉头，不是整个 policy 的可训练参数倍数。不能把结果当作只替换 backbone 的受控实验。

补充闭环结果：同一个DINO FM token主权重，20-step为 **2/100**；两个成功种子在相同local machine、相同专家验证记录与初始输入的4-step对照中均失败。它提供了局部的步数敏感证据，但没有恢复与B0的整体差距，也不能将不同运行流程下的0/100与2/100直接当作无其他差异的严格消融。第8节列出全部口径和限制。

这些事实提出了候选解释，尚未建立因果关系。更大的冻结预训练 backbone 配上更大的随机视觉头，在同一小数据集及固定训练预算下，没有保证一定提高闭环成功率。

另外发现一个**已数值复现的通用 EMA 精度问题**：训练将 EMA 模型转成 BF16，后期小幅 EMA 更新会被舍入掉。现有矩阵闭环读的是主权重，因此这不是上述成功率差距的已证实原因，详见第7节；严格重载通过只代表结构兼容，不代表 EMA 平滑质量正常。

## 1. 已有结果与比较范围

beat：同一组 100 个专家验证种子，FM 4-step、MeanFlow 1-step；PyTorch eager BF16、KV 开、Graph 关。

| Backbone | FM token | FM AdaRMS | MeanFlow token | MeanFlow AdaRMS |
|---|---:|---:|---:|---:|
| EfficientViT B0 | 52/100 | 47/100 | 25/100 | 28/100 |
| EfficientViT B1 | 50/100 | 62/100 | 30/100 | 41/100 |
| EfficientViT B2 | 31/100 | 47/100 | 14/100 | 39/100 |
| DINOv3 ConvNeXt-Base | 0/100 | 19/100 | 4/100 | 23/100 |

这组结果存在 DINO 明显偏低的现象，但也不是“越轻量越好”的单调关系：B1 AdaRMS 高于 B0。handover 已完成的 FM token 结果是 Base 3/100、B0 1/100；FM AdaRMS 则是 Base 9/100、B0 22/100。故应分别讨论任务、注入方式与算法。

两个数据集各有 50 条 episode，实际划分为 **49 条训练、episode4 留出验证**。beat 全部是 200 epoch；handover 原先开始的 5 组保持 200 epoch，后来未开始的 11 组改成 300 epoch。后者完成后必须标注训练预算，不能混称同预算比较。

## 2. 权重、源码和配置核查

检查实际运行的 beat、handover 冻结源码。训练器、数据集、FM/MeanFlow policy、视觉管线、DINO 编码器、DiT blocks、eager 推理和闭环部署适配器这 8 个关键文件，两批实验的 SHA256 全部相同，未发现 handover 对 DINO 暗中换了一条实现路径。完整哈希保存于本次审计目录的 `source_hashes.json`。

已完成的 7 个 DINO 主权重（beat 4 个、handover 3 个）逐项核查：

- 每个 backbone 均有 342 个 state tensor、87,566,464 个参数；无缺失键、无非有限值。
- 与实际使用的本地预训练文件逐 tensor 比较，**全部精确相同**；这些主权重里的 backbone 是 FP32。结合训练解析配置与梯度检查，符合冻结预训练 backbone 的预期，排除了训练时漏载成随机权重、误更新 backbone 的情况。
- 本地预训练解析到 `/DRoboticsAILab/users/xianbao01.hou/pretrained/dinov3-cnn-base`，文件大小 350,302,312 bytes，SHA256：`ec90bd798b5fc5b8e30443796a6c24a7a73e28ad85c6c0ceda78b1d249a694cc`。
- 架构为 ConvNeXt，depths `[3,3,27,3]`、hidden_sizes `[128,256,512,1024]`，drop_path_rate=0。不是 ViT 的寄存器 token 结构。
- 官方 Hugging Face 文件目录与模型尺寸可查询，但 gated 文件接口没有提供可用的公开原文件 SHA。因此“等于本地预训练”已经验证，**“本地文件与官方原文件的字节级同一性”尚未独立完成**；不能把前者写成后者。

官方架构参考：[facebook/dinov3-convnext-base-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-convnext-base-pretrain-lvd1689m/tree/main)。历史项目名、目录名中的 `cnn-base` 不代替架构配置核查。

## 3. 数据、预处理和时序

| 检查项目 | 实际测试与结果 |
|---|---|
| 动作语义 | beat 5732 行、handover 14134 行全部比较 `action` 与 `observation.state`，最大差值 0、无非有限数；均为当前 joint14 数据的绝对关节轨迹 |
| normalizer | beat 16 个、handover 当前 7 个已完成主权重，同任务的 normalizer 与 static_joint_mask 完全一致 |
| 颜色与缓存 | 每任务测试 3 个索引、3 路视图；实际未缓存读取、实际解码 episode 缓存、直接 PyAV 解码后 BGR→RGB 三者逐像素相同，重复读取未发生二次翻转 |
| 帧/动作对应 | anchor=i 时，单图观测取 i-1；状态及训练轨迹从 i-1 开始，输出未来动作从 i 开始；采样核查与 parquet 对齐，未发现一帧错位 |
| 图像范围 | uint8 → float [0,1] 一次；processor do_rescale=false，避免重复除以255；ImageNet mean/std |
| 几何预处理 | 240×320 → 256×320，与 rtpolicy、DRRM 几何函数逐像素相同；训练 processor 与 checkpoint 重建 processor 输出精确一致 |
| DINO token | HF 官方类输出 `[3,81,1024]`；81=1 个聚合 token+80 个空间 token。当前去掉首 token 后得到三视图各8×10空间网格，未错误删除空间 patch |
| 官方类/封装一致性 | FP32、BF16 下，S1 wrapper 与相同本地权重的 HF AutoModel、DRRM CNN wrapper 空间输出精确相同 |

缓存与帧对齐是针对所选索引的数值测试，不冒充对所有视频帧的穷举。闭环 SAPIEN 输入按 RGB 处理，不再 BGR→RGB。当前 JPEG roundtrip 不交换通道；已作为独立输入扰动进行对照。

## 4. 视觉管线、训练状态与梯度

融合与压缩使用相同的结构模板：融合深度1、16头、MLP ratio2，压缩为2个ResBlock；内部宽度B0为128、DINO为1024，最终都输出每视图4个1024维token。宽度增大使主要线性/卷积权重近似按维度平方增长；参数差异来自维度设置，并不是发现两者用了不同种类的融合或压缩模块。

当前路径：三视图 RGB → backbone（每视图80个空间 token）→ 多视图融合 → 两级 ResNet 压缩 → 每视图2×2 token、每 token1024维 → scene_adaptor → 12个2048维条件 token → DiT。

| 模块 | EfficientViT B0 | DINO ConvNeXt-Base | 状态 |
|---|---:|---:|---|
| 视觉 backbone | 682,608 | 87,566,464 | 冻结 |
| 多视图融合 | 265,600 | 16,804,864 | 可训练 |
| 视觉压缩 | 756,224 | 40,907,776 | 可训练 |
| 上两项合计 | 1,021,824 | 57,712,640 | 宽度随 backbone 变化 |
| scene_adaptor | 6,295,552 | 6,295,552 | 相同 |
| state_adaptor | 8,423,424 | 8,423,424 | 相同 |
| FM token DiT（含tau embedding） | 344,784,910 | 344,784,910 | 相同 |
| FM AdaRMS DiT（含动作时间融合及调制） | 667,899,918 | 667,899,918 | 相同 |

对 beat 的 B0/Base × 两种 FM 注入，以及 handover B0 FM AdaRMS、Base 两种 FM，共 7 个已完成 policy 做真实 batch 反向传播：backbone 无梯度，视觉融合、压缩、scene_adaptor、state_adaptor、DiT 的参数梯度均非零且有限。只在隔离进程里计算梯度，没有 optimizer 更新或写回权重。训练器默认 optimizer 包含模型参数；本次解析配置没有遗漏视觉头的专用分组。

所有 7 个 policy，训练/eval 模式下视觉条件输出精确相同；当前 DINO drop_path=0，未观察到冻结 backbone 因 train/eval 模式造成不同结果。4 个 DINO policy 的完整视觉输出，加载同一 checkpoint 张量到 rtpolicy 原实现后，BF16 逐项相同。

注意：推理 FP32 对照是将已训练权重以 FP32 运算，不能恢复训练时 BF16 参数更新可能丢失的精度，也不等同于重跑 FP32 训练。

## 5. 推理数值对照与开环证据

7 个 FM policy × 每权重1条训练 episode0、1条留出验证 episode4 × 每条4个 anchor，共56个窗口。各窗口固定相同噪声，对照：BF16 4-step、关闭KV、FP32、10-step、20-step、黑图、未翻转的原BGR、JPEG、左右视图交换，共504次预测。原始指标保存在 `runtime.json`。

- **56个窗口的 KV开/关输出均逐项相同。** 缓存仅在一次采样内复用同层视觉 K/V，未跨观测、跨chunk复用；训练不使用该缓存。
- FP32 或增加到10/20步未在这些窗口上呈现 DINO 一致获益，不能据此断言正式4-step就是低成功率根因；反过来，这个小样本也不能排除其对其他种子的影响。
- 黑图会明显改变预测，视觉分支没有被完全忽略。单个扰动的误差变好或变坏不能用来认定颜色/视图顺序应当修改。

此前已完成的整条单episode开环（episode4、原开环固定 stride/noise 口径）给出下列关节 RMSE，单位为弧度；该指标只针对12个关节，夹爪另计：

| beat 权重 | 训练episode0 RMSE | 留出episode4 RMSE |
|---|---:|---:|
| B0 FM token | 0.016805 | 0.035874 |
| DINO FM token | 0.016523 | 0.021839 |
| B0 FM AdaRMS | 0.014412 | 0.024144 |
| DINO FM AdaRMS | 0.015722 | 0.017906 |

这与 DINO 的闭环排序不同。**专家轨迹上的开环拟合不差，并不保证在自身动作造成的状态分布中稳定。** episode4 是同一50条demo中的留出验证集，不是额外独立测试集。新审计的4-anchor噪声/采样口径不同，不能与该整episode数字拼接成统一指标。

## 6. 基线对照的边界

rtpolicy、DRRM 在这里作为实现参照；用户已说明没有现有训练权重，故没有把它们视为此次任务已经验证过的高成功率基线。

- rtpolicy 的 DINO CNN-Base 配置与当前 S1 使用相同的核心视觉结构。其当前默认训练配置为300epoch、state history3、image history1、horizon8；当前矩阵是state/image history1、horizon24。这里只比较了配置默认值，不能推断用户当时实际运行的解析配置。
- DRRM `dinov3cnnl_3v.yaml` 是ConvNeXt-Large、1536维、3级压缩、融合MLP ratio2；当前Base是1024维、2级压缩。DRRM `dinov3b_1v.yaml` 则是ViT-Base、768维、单视图，属于另一条视觉路径。
- DRRM 当前 `policy_config/dit_fm.yaml` 默认800epoch、horizon46、3帧观测、43个未来动作、FM10步；当前矩阵为200/300epoch、24=1+23、FM4步。不能据“旧项目用过DINO且效果不错”推定它在本次预算和输入结构下理应优于B0。
- DRRM、rtpolicy 的标准FM训练方向与S1一致：noise→action、velocity=action-noise，观测前缀夹持；闭环均同步生成chunk后逐个执行绝对qpos。没有因为DINO推理耗时更长而在异步控制线程里漏执行动作的路径。
- 把本次 DINO FM token 已训练的 DiT 张量严格加载到三个项目各自的原始标准FM head，固定相同输入，在t=0/250/500/750、FP32和BF16共8种条件下，rtpolicy、DRRM对S1的输出差值均为0。DRRM测试直接加载真实head与相对blocks源码，绕过其包初始化对未使用VGGT的依赖，没有用替代attention或空实现。该测试核对实现一致性，不是重现DRRM历史训练效果。

## 7. MeanFlow 与 EMA 补查

额外检查 beat 的 B0/DINO MeanFlow token 与 handover DINO MeanFlow token：实际NFE=1，KV开关逐项相同，真实MeanFlow损失可反传，冻结backbone无梯度，融合、压缩、scene、state、DiT各组梯度有限且非零。解析选项为连续t/r、equal_time_ratio=0、time_embed_scale=1、guidance_scale=1，没有t=r的显式混合分支。

对6个权重、每权重episode0/4各一个anchor，固定噪声比较主权重与EMA，12个窗口中EMA的关节RMSE均更高。这是小样本诊断，不能替代EMA的完整100种子闭环。

沿代码定位到：`src/100hzs1/trainer/train.py` 将 `ema_policy_model` 转为训练 `weight_dtype`（本次BF16）；`src/100hzs1/common/ema_model.py` 在这个BF16副本上执行 `mul_(decay)`、`add_(source, alpha=1-decay)`，没有FP32累积副本。

用beat DINO FM token实际主/EMA张量及末期decay≈0.99889665复算一次更新：

| 抽查参数 | 主权重dtype | EMA dtype | BF16公式更新后改变元素比例 | FP32公式更新后改变元素比例 |
|---|---|---|---:|---:|
| tau embedding首层weight | BF16 | BF16 | 0% | 46.46% |
| scene_adaptor首层weight | FP32 | BF16 | 0% | 99.94% |
| state_adaptor首层weight | FP32 | BF16 | 0% | 99.73% |
| 视觉fuser首层linear1.weight | FP32 | BF16 | 0% | 99.96% |

相同decay下，旧值1、新值1.1的标量，BF16两步公式结果仍为1，FP32结果为1.0001104。以上说明已观察到更新被舍入的机制，并非宣称全程所有EMA参数完全停止。原始结果见 `ema_precision.json`。

建议未来训练用FP32 EMA副本累计，再按需转换推理精度。**本轮没有更改正在运行的训练pipeline、已有EMA或主权重，也没有把EMA切换成正式评测权重。** 已结束训练的FP32 EMA轨迹不能仅靠最终checkpoint恢复；不能把最终主权重转换成FP32就声称修好了历史EMA。

## 8. 闭环定位实验

使用beat种子100003，原100种子中B0 FM token成功、DINO FM token失败的一个实例。以下四项均为独立local machine诊断，PyTorch BF16、KV开、Graph关；每次重置到相同场景，初始关节状态和三路图像逐项相同，固定同一推理噪声生成规则。

| 权重 | 真实NFE | 每次执行动作数 | 成功 | 环境动作数 | 推理调用数 |
|---|---:|---:|---|---:|---:|
| B0 FM token | 4 | 23 | 是 | 96 | 5 |
| DINO FM token | 4 | 23 | 否 | 400 | 18 |
| DINO FM token | 20 | 23 | 否 | 400 | 18 |
| DINO FM token | 4 | 4（只执行预测的前4个，再重新观测） | 否 | 400 | 100 |

B0视频可见抓锤并完成击打，DINO没有完成抓取并持续移动。原始4-step/23动作条件下，前4个chunk“首个预测动作与当前状态”的12关节RMSE（rad）为：B0 `[0.01155,0.03312,0.00473,0.01366]`，DINO `[0.04356,0.02393,0.08166,0.05225]`；后续DINO出现更大的动作跳变。该观察符合闭环偏离专家轨迹后不能稳定恢复的现象，**尚不能判断偏移最初来自哪个训练因素**。

增大积分步数、缩短执行chunk没有修复这个实例；不能推论它们对全部种子无效。四次单seed诊断不能当作新的成功率，也不替换原16组结果。完整记录在 `closedloop_diagnostic_v3.json`、`closedloop_trace_summary.json`，视频与逐chunk输入/预测位于 `closedloop_diagnostic_v3/<case>/`。

20-step确实降低了这个实例的后续动作跳变：例如前4个chunk内最大单关节相邻动作变化，从4-step的 `[0.1193,0.1497,0.2494,0.3013]` rad变为20-step的 `[0.1051,0.0779,0.0469,0.0638]` rad。平滑改善在这个单种子上没有转化为任务成功；全种子诊断结果如下。

### 同100种子20-step诊断与配对4-step对照

同一个DINO FM token主权重，固定原beat的100个种子，已在local machine L4完成全部100条策略episode。保持PyTorch eager BF16、KV开、Graph关、RGB不翻、JPEG roundtrip、23动作chunk。原种子文件SHA256为 `fca2035b8f623d52f980843f8f578710c520997db5c7f334f557ebec5a2ec8da`；实际journal逐条核对100个唯一seed与完整原清单一致，成功数重新计数为2。

| 对照 | 真实NFE | 种子范围 | 成功 / 完成 | 说明 |
|---|---:|---|---:|---|
| 原正式DINO FM token结果 | 4 | 原固定100个 | 0/100 | 原多卡L4运行，结果保留 |
| 本次local诊断 | 20 | 同100个 | **2/100** | 独立诊断，包含下述专家重试与验证记录复用流程 |
| 同机配对4-step复跑 | 4 | 仅20-step成功的2个 | **0/2** | 相同local环境、专家记录及初始输入；不是新的100种子成功率 |

| 种子 | NFE20 | NFE20动作数 / 推理调用 | 同机NFE4 | NFE4动作数 / 推理调用 |
|---|---|---:|---|---:|
| 100061 | 成功 | 76 / 4 | 失败 | 400 / 18 |
| 100073 | 成功 | 198 / 9 | 失败 | 400 / 18 |

两次配对4-step复跑均严格匹配相应专家初始关节状态与全部4路原始相机RGB哈希；20-step对这两项的相同检查也通过。实际模型使用三路视图，额外校验第四路相机用于场景一致性。日志逐seed确认eager、BF16、NFE20或4、action_steps23、Graph关闭；NFE20及配对NFE4的最终评测进程正常退出，最终journal为completed，Graph captures/replays均为0，102段视频非空。实际环境为Torch2.7.1+cu126、SAPIEN3.0.0b1、NVIDIA L4。

重新计算checkpoint主文件与配置SHA256，并与此前量化时保存的manifest比较，均一致：`model.safetensors` 为 `9ae8588166a5f90d70a190e87809a0e30e9e6346b7caa37cbe5f4219433b99d6`，`config.json` 为 `7cab2eac5b262436112c1e1f9fbae8525651bd9d03754e1304a89255f0fc7fa9`。未切换EMA、修改权重或替换模型。

**本次结果支持：这两个实例在所执行的对照中对积分步数敏感；把该权重增至20-step仍只有2/100，没有恢复原B0 FM token的52/100水平。** 两个实例各仅复跑一次，且按20-step成功结果选取，不构成无偏的额外4-step成功率估计，也不足以证明确定性复现、统计显著提升或步数是整体差距的唯一原因。原正式4-step与本次20-step还存在运行主机及专家验证组织方式差异，不能忽略后直接作因果推断。

### 专家复检中断、恢复与验证流程差异

原清单中的100005、100016、100019在专家复检中曾失败，均发生在相应学习策略episode之前，没有把专家失败计为策略失败，也没有更换、跳过种子。新进程中100005、100016后来通过；100019的6次独立探针中，不加载policy为1/3通过，加载DINO为2/3通过。初始物体位置和运行标志相同，不能据此归因于DINO加载，专家重复执行的稳定性原因仍未查明。

环境源码71项、task_config18项、description619项与原beat快照一致，完整场景identity为 `ed3c4d4ab844704fe00ba868381a029a18ee2b6deb4183c3112d9a6dc6396c2d`，匹配原种子注册表。前16条已完成策略episode全部保留。为完成剩余84条，先用不加载policy的独立进程对原84个种子逐一执行实际专家检查，必须同时规划成功与任务成功，保存专家info、初始关节状态和全部4路原始RGB哈希。全部84条实际通过，未按学习策略表现筛选或换seed。

后续策略进程复用这些当前主机上实际通过的专家记录，并在每个episode的第一次推理前断言关节及全部相机原始RGB精确相同；84/84通过。独立验证前的只读get_obs也核查不改变物体位姿。两个成功种子的NFE4对照使用同一批记录，2/2初始输入匹配。**这改变了本次诊断中专家验证与学习策略执行的组织方式；它不是对冻结源码或正式pipeline的修复，也不能宣称与原多卡正式评测整个执行协议完全相同。** 初始观测一致并不证明所有隐藏仿真状态或接触求解都逐步确定，因此配对结果仍需按上述限制解释。

原始失败日志、恢复日志和journal历史failed事件全部保留；最终核验采用完整100个实际策略结果、最终completed事件和正常退出，不抹除历史中断。完成证据在 `closedloop_nfe20_100_v1/`、`closedloop_nfe4_success_controls_v1/`、`expert_cache_v1/`，独立审计汇总为 `nfe20_and_paired_nfe4_final_audit_20261031.json`。

## 9. 后续决策与证据归档

正式16组继续按已有队列执行。当前没有证据支持直接翻转DINO颜色、关闭KV或替换预训练路径作为“修复”；不做会混淆进行中实验的静默更改。

下一步应另建单变量受控实验，而不是据本轮审计直接改正在运行的训练：

1. 固定DiT、flow/注入、数据划分、观测历史、优化器与更新步数，将两种backbone的输出先映射到同一视觉头宽度，再使用相同融合、压缩结构；输入投影因backbone维度不同产生的参数单独列出。这样可检查视觉头容量是否影响差距。
2. 独立比较更多观测历史或执行chunk设置，不能把历史项目默认值当成已证实更优的配置；每项保留相同训练预算和完整同100种子闭环。
3. 将FP32 EMA累计作为独立训练改动，并固定其他变量、多训练seed核查主权重和EMA。它是已确认的通用实现问题，但仍不能倒推为当前主权重DINO低成功率的原因。

以上是后续实验方案，本轮没有启动额外正式训练，也没有修改当前训练pipeline。视觉头容量、训练预算与闭环分布偏移仍是候选因素，尚未被消融实验确认。

审计脚本、日志、原始JSON、逐chunk状态和视频全部位于：

`/DRoboticsAILab/users/licheng.yang/100HzS1_tmp/dino_audit_20261031_v1/`

报告放入 `/data/vepfs/licheng.yang/100HzS1/docs/`。没有临时权重放入源码工作树的runs，没有覆盖原checkpoint；失败测试启动的日志也保留在审计目录，未把启动失败计作模型失败。初次测试遇到的脚本导入路径、headless渲染环境配置已在独立测试进程修正，正式任务未受影响。
