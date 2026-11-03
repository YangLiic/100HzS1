# B0六组：GR00T式归一化、无池化、标准FM4

本次六组全部从头训练一次；旧权重、旧结果与冻结快照保留。

| 编号 | 配置 | repeat stage4 | 视觉映射 | DiT宽度 | 成功数/100 | 参数量 | W&B实际步数 |
|---:|---|---|---|---:|---:|---:|---|
| 1 | stage4_resmlp_no_norm | 是 | Linear128→1024；残差MLP1024→2048→GELU→1024，无RMSNorm | 1024 | 51 | 93,817,216 | [finished 8771](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/zrxwn207) |
| 2 | stage4_rms | 是 | Linear128→1024；RMSNorm＋残差MLP1024→2048→GELU→1024 | 1024 | 55 | 93,818,240 | [finished 8771](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/2aaemqnp) |
| 3 | mlp1024 | 否 | Linear128→512；普通MLP512→1024→GELU→1024 | 1024 | 44 | 91,744,640 | [finished 8771](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/yy0onj8d) |
| 4 | rms | 否 | Linear128→1024；RMSNorm＋残差MLP1024→2048→GELU→1024 | 1024 | 17 | 94,434,176 | [finished 8771](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/5tp4q7sn) |
| 5 | stage4 | 是 | Linear128→1024；普通MLP1024→2048→GELU→2048 | 2048 | 43 | 360,605,056 | [finished 8771](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/48hulw7r) |
| 6 | stage4_mlp1024 | 是 | Linear128→512；普通MLP512→1024→GELU→1024 | 1024 | 46 | 91,128,704 | [finished 8771](https://wandb.ai/gonzalezmonica966-digua/100HzS1/runs/67gc1tcv) |

共同设置：beat_block_hammer/demo_clean、50demo（49训练/1留出）、3视角、seed42、200epoch/8771步、batch128、BF16、标准FM/tau_token/1000桶、推理4steps[0,250,500,750]。每视角自然2×3=6token，共18视觉token，全部不池化。DiT八层，head_dim64，16/32heads，内部FFN始终D→D→GELU→D。

GR00T式归一化：只用选定训练episode，state和action分别计算q01/q99，映射并裁剪[-1,1]，常量维归零，最终预测逆变换也裁剪。图像仍ImageNet标准化。旧minmax实验不冒充本次结果：其中部分旧组还有2×2池化差异，不能把两轮全部差异只归因于归一化；不同归一化下训练loss不直接横比。

共享stage4组：B0预训练冻结；同一冻结stage4额外调用两次、一个共享可训练1×1Conv128→64桥接（8192参数），80→20→6tokens/view后融合，无原两个ResBlock。未repeat组：80tokens/view融合后通过原两个ResBlock到2×3，无池化。融合末LayerNorm保持；视觉RMS仅在明确rms组启用。

CPU真实数据预检、八卡两次真实forward/backward零optimizer启动检查、初始与最终main/EMA严格重载、实际4次FM采样、18token/KV、2000→4000模块更新及每份186个B0冻结张量按保存精度匹配预训练均已验证。无蒸馏、教师、额外基线或NFE扫描。

闭环：最终main、eager/BF16/NFE4/KVon/GraphOff/RGB不翻转；每组四分片×25，相同100固定唯一seed及100非空唯一视频。

原runner同seed场景恢复事件总数32，事件详见report_supplement.json；没有替换seed或因policy失败重跑挑结果。

冻结commit `59ba1411f28a2b80d7df18bcc07f899dd2238f1a`，974个源文件/配置哈希；两个request成功且两机holding。六组W&B实际服务端均FINISHED8771，CPU relay source/queued/remote均补齐。

每组只有一个训练seed，不能宣称为多训练seed稳定均值。

## 产物路径

- **stage4_resmlp_no_norm**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/stage4_resmlp_no_norm`；闭环 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/stage4_resmlp_no_norm/main_nfe4_100_v1/result.json`；最终loss 0.00351。
- **stage4_rms**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/stage4_rms`；闭环 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/stage4_rms/main_nfe4_100_v1/result.json`；最终loss 0.00358。
- **mlp1024**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/mlp1024`；闭环 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/mlp1024/main_nfe4_100_v1/result.json`；最终loss 0.00275。
- **rms**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/rms`；闭环 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/rms/main_nfe4_100_v1/result.json`；最终loss 0.00174。
- **stage4**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/stage4`；闭环 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/stage4/main_nfe4_100_v1/result.json`；最终loss 0.00252。
- **stage4_mlp1024**：训练 `/DRoboticsAILab/users/licheng.yang/checkpoints/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/stage4_mlp1024`；闭环 `/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1/beat_block_hammer/beat_b0_groot_nopool_six_fm4_20261102_v1/stage4_mlp1024/main_nfe4_100_v1/result.json`；最终loss 0.00741。

审计：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_groot_nopool_six_fm4_20261102_v1/final_audit.json`；补充：`/data/vepfs/licheng.yang/100HzS1/runs/beat_b0_groot_nopool_six_fm4_20261102_v1/report_supplement.json`。

## W&B网络恢复说明

第5组日志转发的SSH隧道中途断开，服务端曾因日志心跳中断将run标为crashed；GPU训练和闭环均正常完成。2026-11-02 18:25恢复专用隧道后，在原run `48hulw7r` 补齐6618→8771步并正常finish；18:27独立API核验全部六组FINISHED8771。没有新建run或重跑GPU实验。CPU审计器因旧SSL重试阻塞，核对argv/cwd后仅重启本次CPU watcher/probe，18:28完成最终审计。
