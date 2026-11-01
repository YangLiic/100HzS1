# Beat B0 双独立可训练 MBConv：最终结果

machine2 本次最终主模型的闭环结果为 **13/100（13%）**，四分片分别为1/25、4/25、3/25、5/25。之前共享stage4重复方案为35/100；这一次独立可训练MBConv没有改善该任务的成功率。两者只有一个训练seed，本结果不能证明所有MBConv压缩设计都较差。原2×2压缩、默认2048宽度的独立对照已按用户后续要求在machine1运行，由另一个巡检负责。

## 实际模型

原EfficientViT-B0严格加载预训练并冻结，包括BN统计。240×320图像按既有预处理到256×320，输出每视图128×8×10（80token）。新增两层独立可训练MBConv stride2，空间依次4×5、2×3（20、6token）。每层为128→512的1×1升维、3×3 depthwise stride2、512→128的1×1降维。分别复制stage4两个local MBConv的预训练参数作为初始化，不共享参数，不复制LiteMLA或旧残差支路，无128→64桥接。

以每视图6token/128维进入多视图融合，随后Linear128→1024、scene MLP1024→2048→GELU→2048。没有原ResNetAdapter或额外池化。三视图共18条件token；DiT/state/action/t/r保持2048，8层32heads，每层K/V为[B,32,18,64]，原始/输出动作仍14维。

## 训练及严格核验

B0、beat_block_hammer、50demo、seed42、200epoch、globalbatch128、8×L4、BF16、MeanFlow/tau_token、minmax、horizon24、离线BGR→RGB；不恢复之前完整策略权重。正式8771步完成，最终loss0.0234，报告训练时长49分52秒。训练日志未见非有限loss、OOM或异常退出。

- main和EMA最终严格重载，有限动作[1,23,14]、条件[1,18,2048]。
- 原冻结B0按checkpoint保存精度与预训练精确匹配；main为FP32，EMA为BF16。
- 两新增MBConv在main/EMA中均各有9/9浮点张量相对初始化更新；新BN统计已更新，EMA BN buffers与main同步。
- 原源码980文件、runner、训练配置、预训练文件、main/EMA及固定seed哈希重新验证全部一致。
- 最终main闭环：MeanFlow NFE1、eager BF16、KV开、Graph关、SAPIEN RGB不翻转。
- 四分片25seed各自完成；100固定唯一seed精确覆盖，100独立非空视频逐项核验，成功数按逐回合记录重算13。原生expert初始化重试保留相同seed，未替换seed。
- 唯一request成功exit0，于2026-11-01 17:52:51（Asia/Shanghai）结束，machine2已恢复holding。
- W&B服务端实际8771步、state=finished。[训练曲线](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/efewj7ra)。末次复查时旧CPU网络代理不可用，已恢复仅供W&B使用的转发，再次独立API确认同步完整；M2的正式日志此前已上传结束。

首次v1只在smoke验证遇到FP32/BF16逐位比较误报，未启动正式训练；v1产物保留。针对新训练BN补齐EMA buffers同步后，以独立v2路径重新smoke并完成正式实验，未修改运行中的源码或覆盖旧产物。

## 路径和版本

- 权重目录：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_mbconv2x_2x3_20261101_v2`；main为model.safetensors，EMA为ema/model.safetensors。
- 闭环汇总：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_mbconv2x_2x3_20261101_v2/main_nfe1_100_v1/result.json`，同目录shard0–3下保存逐回合视频。
- 冻结源码、配置、调度和审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_mbconv2x_2x3_20261101_v2`；解析配置configs/train.yaml，final_audit.json/final_audit_mbconv.py记录完整终验。
- 源码commit：`316775f06dff1137d717557ead91f2912b67c544`。
- main SHA256：`e40f90eedd95765ffe428755decf1ec7800c456fa34653feda2f22b8796dba6c`。
- EMA SHA256：`c1fbe463392748c3f5c9d741557517a6ad947d2a8f21ba17623c6db1eb684532`。
- seed SHA256：`fca2035b8f623d52f980843f8f578710c520997db5c7f334f557ebec5a2ec8da`。
- 训练配置 SHA256：`1afc94af2141fe056c3f13905af38ad8fd9d21f39bf3534f962870798f2a3af1`。

没有追加实验、EMA评测或NFE扫描，没有push。machine1的默认维度对照及其巡检保持独立。
