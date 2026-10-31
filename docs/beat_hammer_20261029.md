# beat_block_hammer：16 组闭环评测

每组 100 个相同固定种子；数值为成功次数。标准 FM 使用 4-step，MeanFlow 使用 1-step；PyTorch eager、BF16、CUDA Graph 关闭、KV cache 开启。

| Backbone | FM + tau_token | FM + tau_AdaRMS | MeanFlow + tau_token | MeanFlow + tau_AdaRMS |
|---|---:|---:|---:|---:|
| B0 | 52/100 | 47/100 | 25/100 | 28/100 |
| B1 | 50/100 | 62/100 | 30/100 | 41/100 |
| B2 | 31/100 | 47/100 | 14/100 | 39/100 |
| DINOv3 Base | 0/100 | 19/100 | 4/100 | 23/100 |

训练采用 50 条 demonstration、200 epoch、全局 batch 128、8 卡、冻结预训练视觉 backbone、训练视频 BGR→RGB；MeanFlow 使用连续 t/r、equal_time_ratio=0、guidance_scale=1.0、无 Dispersive Loss。

所有权重的主模型和 EMA 均已严格重载；结果核对实际 100 种子集合、成功数、后端、精度、步数、Graph、KV cache 及 checkpoint。随机种子一致并不能消除单次训练与评测的统计波动。

FM 是 4-step，MeanFlow 是 1-step；算法与推理步数同时不同，结果不能单独归因于算法。DINOv3 Base 的 FM + tau_token 为 0/100，是本次评测的实际结果；这不单独说明原因。

各组 checkpoint 和结果文件路径见同目录 results.json。
