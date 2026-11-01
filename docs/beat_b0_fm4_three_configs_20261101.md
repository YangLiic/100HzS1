# B0标准FM4：三种维护配置

| 配置 | 池化位置 | Linear投影 | 视觉MLP | DiT／状态／动作／时间宽度 |
|---|---|---|---|---:|
| 默认baseline | 融合→两个ResBlock后，池化2×2 | 128→1024 | 1024→2048→GELU→2048 | 2048 |
| 融合前共享stage4 | stage4额外两次后、融合前池化2×2 | 128→1024 | 1024→2048→GELU→2048 | 2048 |
| 视觉侧RMSNorm残差降维 | 保留默认压缩和池化 | 128→1024 | u＋Linear2048→1024(GELU(Linear1024→2048(RMSNorm(u)))) | 1024 |

三种配置最终都是每视角4token、三视角12token。
Linear输出1024只是视觉MLP的输入维度，并不自动等于进入DiT的最终宽度。
默认组/压缩组的视觉MLP输出2048，因此状态/动作/时间token也为2048。
RMS残差组视觉MLP输出1024，因此状态/动作/时间token为1024。

默认：冻结B0的128×8×10→三视角融合→两个可训练ResBlock（4×5→2×3）→平均池化2×2→Linear→视觉MLP。
压缩：冻结B0的128×8×10→共享128→64桥接＋同一个冻结stage4，额外两次得到4×5→2×3→平均池化2×2→三视角融合→Linear→视觉MLP。
压缩组的池化在融合前，增加0个可训练参数；没有额外独立MBConv，也没有融合后共享stage4。
RMS组的直连旁路不经过RMSNorm，保持u＋MLP(RMSNorm(u))。

本次只整理YAML、测试和说明文档，所有src实现（包括DiT）哈希保持不变。
沿用已恢复的DiT结构：8层、head_dim64、等宽FFN D→D→GELU→D。
RMS组用原有hidden_size=1024/16heads配置；另两组2048/32heads。没有新增FFN宽度选项。
状态和带噪动作仍共享原三Linear＋两GELU，14→D→D→D；时间仍是原频率编码/MLP→D，FM没有r。

从原始输入角度，14维状态和标量时间使用1024维嵌入是合理的试验选择；不能只凭输入维度低保证完整DiT降宽无损。
这里的D同时决定注意力、FFN和动作去噪的计算容量，并非仅存储状态/时间的空间。
不改变DiT实现时，各路token进入主干必须保持同一D；本次不增加跨维度投影支路。

共同：beat_block_hammer/demo_clean，50demo、seed42、200epochs、batch128、BF16、三视角、minmax、标准FM/tau_token、1000桶、4steps。
保留原数据颜色/缓存设置，推理eager/KV-on/Graph-off/RGB不翻转。
只维护默认与两个消融，不启动新训练/评测，不处理旧权重、结果或冻结快照。
旧直接降维和6token YAML移到docs/archived_configs/beat_b0_fm4_20261101/，不再是Hydra实验入口。

验证完成：34项CPU测试通过。三种完整实际模型的条件/状态/时间/KV维度、池化顺序和等宽FFN均核验通过。
全部108个src Python源码哈希保持不变；本次没有改动DiT或任何模型实现。
详细核验：runs/b0_three_configs_20261101_v1/verification.json。
