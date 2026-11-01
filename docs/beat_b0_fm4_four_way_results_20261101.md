# B0标准FM4：四组最终结果

> 清理记录：用户于2026-11-01明确要求退役“融合后共享stage4＋2×2”组。
> 该组专属源码分支/YAML、正式与临时权重、RoboTwin结果及100个视频、独立run目录已删除。
> 本文保留其8/100历史统计和W&B日志链接；该组原路径已失效。其他三组及RMSNorm实验保留。

四组均已完成训练、最终main/EMA严格重载、100固定seed闭环与完整审计。

| 实验 | 压缩方式 | DiT维度 | token/视角 | 总参数量 | 成功数/100 |
|---|---|---:|---:|---:|---:|
| 原始baseline | 原融合后两个ResBlock+2×2池化 | 2048 | 4 | 361,208,588 | **52/100** |
| 降维组 | 原融合后两个ResBlock+2×2池化 | 1024 | 4 | 91,738,380 | **22/100** |
| 融合前共享stage4 | 融合前共享stage4额外两次；无末端池化 | 2048 | 6 | 360,604,940 | **48/100** |
| 融合后共享stage4+2×2 | 融合80token/视角后替换两个ResBlock；保留2×2池化 | 2048 | 4 | 360,592,652 | **8/100** |

原始baseline本次52%，降低到1024维的版本22%，融合前共享stage4版本48%，融合后共享stage4且保留2×2池化版本8%。最后一组分片成功数2+3+2+1；本配方下不适合替换原压缩模块。失败机制尚未定位，不能将结果直接归因于池化或某一个算子。融合前版本与baseline相差4回合，单次训练与100回合不能证明小差异稳定存在。

## 相同实验条件

beat_block_hammer/demo_clean，B0预训练冻结，三视角left/right/head，50demo（49训练/1验证、5614/118样本）、训练seed42、200epoch8771优化步、batch128、BF16、minmax、绝对joint14、horizon24（1状态+23动作）。离线BGR转RGB，闭环SAPIEN为RGB且不翻转。

全部是标准FM/tau_token，1000时间桶、4-step；实际main和EMA均验证4次DiT前向[0,250,500,750]，无MeanFlow r分支。闭环最终main，eager/BF16/KV-on/Graph-off，四个25-seed分片、两条独立4GPU热队列lane。同一组100个唯一固定合法seed，每组100个独立非空视频，总计400个视频；未替换seed、未运行EMA对比或NFE扫描。

Seed SHA256: `fca2035b8f623d52f980843f8f578710c520997db5c7f334f557ebec5a2ec8da`。

## 每组产物与日志

### 原始baseline

视觉路径：`128→1024→2048→2048`；三视角共12个条件token。
训练机器：machine1，request `s1-beat-b0-fm4-native2048-20261101-v1`；冻结源码 `4d4b6ada7d80eeb133eecabd444c91a16ede863f`。
结果：52/100，最终loss 0.00115，8771训练步；main和EMA严格重载及冻结B0检查通过。

- 训练权重：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_fm4_native_d2048_baseline_20261101_v1`
- 评测汇总：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_fm4_native_d2048_baseline_20261101_v1/main_nfe4_100_v1/result.json`
- 运行与审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_fm4_native_d2048_baseline_20261101_v1`；`final_audit.json`
- W&B：[正式run](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/3dvbn2u7)，实际服务端 **FINISHED / 8771**。

### 降维组

视觉路径：`128→512→1024→1024`；三视角共12个条件token。
训练机器：machine1，request `s1-beat-b0-fm4-width1024-20261101-v1`；冻结源码 `cc6dc4582debec7e61f15b91bd0e3845a09cf129`。
结果：22/100，最终loss 0.00274，8771训练步；main和EMA严格重载及冻结B0检查通过。

- 训练权重：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_fm4_ablation_20261101_v1/width1024_post2x2`
- 评测汇总：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_fm4_ablation_20261101_v1/width1024_post2x2/main_nfe4_100_v1/result.json`
- 运行与审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_fm4_ablation_20261101_v1/cases/width1024_post2x2`；`final_audit.json`
- W&B：[正式run](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/pt5tbit0)，实际服务端 **FINISHED / 8771**。

### 融合前共享stage4

视觉路径：`128→1024→2048→2048`；三视角共18个条件token。
训练机器：machine2，request `s1-beat-b0-fm4-stage4-20261101-v1`；冻结源码 `cc6dc4582debec7e61f15b91bd0e3845a09cf129`。
结果：48/100，最终loss 0.00204，8771训练步；main和EMA严格重载及冻结B0检查通过。

- 训练权重：`/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_fm4_ablation_20261101_v1/stage4_d2048`
- 评测汇总：`/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_fm4_ablation_20261101_v1/stage4_d2048/main_nfe4_100_v1/result.json`
- 运行与审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_fm4_ablation_20261101_v1/cases/stage4_d2048`；`final_audit.json`
- W&B：[正式run](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/bnjepzv2)，实际服务端 **FINISHED / 8771**。

### 融合后共享stage4+2×2

视觉路径：`128→1024→2048→2048`；三视角共12个条件token。
训练机器：machine2，request `s1-beat-b0-fm4-poststage4-2x2-20261101-v1`；冻结源码 `7edeac83708f955f22c7b72847f18abc36315d13`。
结果（已退役清理，仅保留历史统计）：8/100，最终loss 0.0024，8771训练步；main和EMA严格重载及冻结B0检查通过。

- W&B：[正式run](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/6aapwos1)，实际服务端 **FINISHED / 8771**。

## 融合后版本的实现与训练证据

共享的stage4包含入口MBConv和两个EfficientViT block。原B0及其stage4始终冻结；一个128→64的1×1桥接（8192参数）在两个额外调用和所有视角间共享。路径是80token/视角先融合，再8×10→4×5→2×3，最后原AdaptiveAvgPool2d到2×2、原Linear128→1024。没有两个独立新增MBConv，也没有保留原两个ResBlock。

28项CPU测试、八卡两步真实数据smoke、最终main/EMA严格重载和操作顺序验证均通过。2000→4000正式checkpoint证明main桥接8192元素、融合view_embed384元素发生更新；EMA分别5891、270元素变化。每个checkpoint的main/EMA均有186个冻结B0张量精确匹配其保存精度下的预训练值。推理输出有限，训练无非有限值/OOM/运行时错误。

## 汇总机器标签的校正记录

最终审计发现新组dispatcher在汇总JSON中沿用了字面量`machine1`。实际运行的冻结plan、唯一machine2 request、supervisor及全部eval事件hostname均对应`t-20260819215926-zt9qr-worker-0`。这是汇总元数据标签错误，100回合实际在machine2执行。

原始`result.json`和冻结dispatcher均原样保留，未覆盖。另存`result_machine2_verified.json`仅校正machine字段并附带原件/dispatcher/执行事件哈希、精确Job/request与依据；成功标记、视频、checkpoint、seed等其余全部字段与原件一致。完整审计额外验证该校正；8/100来自逐回合重新计数。

## 收尾边界

四个request均已成功，machine1和machine2恢复holding。全部权重、视频、日志、配置和源码快照保留；未push。用户后续讨论的残差降维/蒸馏/FFN扩张仅为方案，未启动任何新增训练或评测。收尾核验确认没有其他消费者后，专用W&B helper PID81428和SSH转发PID81429已于14:11:28UTC退出；beat-b0-fm4-ablation巡检已删除。收尾凭据位于最后一组run目录的cleanup_preflight.json和transport_cleanup.json。
