# RTPolicy 性能优化：思路与做法

目标：提高真实训练 it/s，同时减少长尾与 SM 波动。先找关键路径，再改一项、复测一项；以端到端收益和训练质量决定是否保留。

全部成功率与效率结果统一收录于 [EFS实验成绩归档](/data/efs/datasets/xianbao01.hou/rtpolicy_archives/20261016_071211/RTPolicy-backbone-compile/docs/backbone_compile_experiment.md)（2026-10-21更新）。

希望先看直观例子，可读飞书[《RTPolicy：Lance训练优化与实验结果》](https://horizonrobotics.feishu.cn/wiki/MqF0wlQ9PixLJHkBVwYcTZW6nnf)中实验后的“性能优化讲解”章。

## 1. 先建立可信的基线

- 固定数据、分辨率、相机数、模型、精度、优化器、每卡 batch、GPU 数和训练阶段。多卡 `it/s × 全局 batch` 才是全局 samples/s。
- 先预热 100 步，正式计时至少 1000 步；CUDA 在计时边界同步。记录平均速度、步骤 P95/P99、最大值，以及每秒 GPU/SM、频率和功耗。
- 统一 validation/checkpoint 设置；长训累计速度保留保存停顿。编译首轮开销与稳态速度分开统计，续训按完整计时区间统计，避免进度条瞬时速度偏差。
- Profiler 单独短跑。它会扰动 CPU 调度和 GPU 发射，不能用插桩后的绝对速度代替正常训练速度。

训练时在另一终端采样，结束后用 Ctrl-C 停止监控；只分析正式计时窗口：

```bash
nvidia-smi dmon -s puc -d 1 -o DT > gpu_sm.log
```

## 2. 按等待位置定位瓶颈

数据路径：EFS → Lance 字节读取 → JPEG/视频解码 → CPU 预取 → H2D → 模型前反向 → AdamW/EMA → 多卡同步。
这些阶段会重叠，逐阶段耗时不能简单相加；一个 rank 卡住，也可能让其余卡等待。

| 现象                     | 先检查什么                          | 本轮做法                                                       |
| ------------------------ | ----------------------------------- | -------------------------------------------------------------- |
| 换 EFS 后变慢、SM 有低谷 | 读取字节数、慢 syscall、worker 队列 | 拆开读取/解码/H2D，不只看 DataLoader 总时间                    |
| 每隔约 30 秒停几秒       | 慢步骤时间与读取栈是否对齐          | 捕获到 EFS `preadv` 最长约 6.9 秒，随后 CPU 队列耗尽、多卡等待 |
| 大多数步骤正常、P99 很差 | 长尾来源，而非盲目增加 worker       | JPEG blob 显式按需 mmap，绕开本轮已观察到的读取停顿            |
| 加载已很短，但训练仍慢   | GPU 热点、细小算子、CPU 发射、同步  | 编译冻结 backbone，再用固定输入判断加载剩余影响                |

观察到的是文件系统读取停顿；其内部为什么周期性触发，尚未用内核证据确认。视频/JPEG 快慢还取决于 codec、GOP、解码器缓存和读取放大，比较时必须对齐输入条件。

## 3. 本轮实际保留了什么

1. **Lance JPEG blob + 按需 mmap**：初始化只取位置/长度描述符，按采样读取 JPEG 范围。`MADV_RANDOM` 避免推测预读，读后 `MADV_DONTNEED` 释放本进程映射页；OS 文件缓存仍由系统管理。文件映射上限 16，只支持已挂载、不可变的 blob-v1 单文件 fragment，显式选错布局会报错。
2. **有界预取**：64 个总 worker / 8 卡 = 每卡 8 个；CPU `prefetch_factor=2`，约 16 个在途 batch/卡；全局 batch256 / 8 = 每卡32。CPU 队列的名义容量约512样本/卡，另有当前批次与单批 CUDA 预取。深度加大只增加缓冲，生产速度不足时不能根治，也会增加内存。
3. **H2D 与计算重叠**：pinned CPU memory、non-blocking、独立 CUDA stream 提前传一批；保留样本顺序及 stream 生命周期。当前实现要求 CUDA、CPU 解码、梯度累积1。
4. **编译冻结 backbone**：Inductor 融合算子，减少中间写回和发射次数；每一步仍计算图像特征，checkpoint 参数键不变，推理可直接 eager 加载。早期同配置短测为2.401→2.937 it/s，约22%，随后才继续解决读取长尾。BF16 结果不保证逐位一致，因此先对账，再做训练/任务测评。
5. **正确恢复续训**：恢复 `checkpoint/ema/model.safetensors` 与 EMA 步数；普通非 fused/noncapturable Adam 的 step 标量放回 CPU，避免逐参数同步；normalizer 恢复后放回模型设备；修正续训进度条。默认随机有放回采样不保证逐样本精确续接。

正常训练的媒体预载预算保持0，直接读取 EFS；固定输入只用于下面的诊断。

## 4. 用固定真实输入判断还值不值得优化加载

只取一个真实 batch 驻留 GPU，关闭 DataLoader，反复执行完整 backbone/前反向/AdamW/EMA；不缓存特征。固定组前后核对输入哈希，加载组首批也核对，避免改了样本却误认为只改了性能。

以下为合入前冻结实验快照的结果：2026-10-16，3路640×480、CNN Base、编译冻结 backbone、BF16；每组100步预热+1000步计时。平台配额标A800，实际节点为8×A100-SXM4-80GB：

| 配置                  |  it/s | host步骤P99 | 步骤CV | SM均值 / 标准差 |
| --------------------- | ----: | ----------: | -----: | --------------: |
| 8卡 EFS mmap 正常读取 | 3.113 |    338.08ms |  1.84% | 96.79% / 2.16pp |
| 8卡固定 GPU 输入      | 3.242 |    322.85ms |  1.33% | 96.39% / 1.51pp |
| 单卡固定 GPU 输入     | 3.407 |    296.73ms |  0.43% | 96.78% / 1.10pp |

两组8卡全局batch256、首批全rank逐字节一致。去掉加载只快 **4.15%**，说明当前加载已跟上训练；差值包含解码、H2D、调度和资源竞争，不能全部当作EFS读盘时间。EFS组最长host步骤375ms，仍有偶发单卡SM低谷。SM标准差汇总各活动卡的1秒采样，包含卡间差异。

单卡全局batch32，首批与8卡rank0不同、形状相同，且归约/scheduler/频率条件不同，仅作参考。独立640长训累计9900计时步为 **3.105 it/s**。此次未清空OS文件缓存，不能承诺TB级冷缓存同速。此前320×240五任务编译/未编译成功率均值均为79.6%，640×480本轮验证的是性能。

**2026-10-21：RTP同机补测640×480 Video（原AV1）与JPEG。**

| 配置        | 训练it/s |  步骤P99 | SM均值 | SM标准差 |
| ----------- | -------: | -------: | -----: | -------: |
| Video未编译 |   2.2896 | 462.25ms | 96.39% |   3.64pp |
| Video编译   |   2.7248 | 642.71ms | 93.21% |   7.36pp |
| JPEG编译    |   3.1061 | 339.79ms | 96.93% |   2.15pp |

同一checkpoint20000、同模型/8×A100/global256/local32，各100步预热+1000步计时；Video编译比未编译快 **19.01%**，Video编译相对同轮JPEG编译的it/s差为 **-12.28%**。两种媒体都直接读EFS，预载预算0，无本地媒体copy；Video为PyAV CPU/file范围读取，JPEG为CPU/mmap。Video两组首批在8rank上逐字节一致。

Video编译组取批次最大等待 1.751 秒，说明编译后输入等待长尾更可见；该耗时含取批次/设备传输及lookahead，不等于纯读盘时间。SM均值不能代替训练速度。

启动、首次编译及预热未计入it/s；本轮先通过短EFS路径的spawn IPC检查，并按rank隔离EFS编译缓存。失败启动未计入结果；Video eager复用同源码/同配方v2完整窗口，两个编译组用v3隔离缓存。本轮只测效率，不测成功率。完整记录：[EFS结果](/data/efs/datasets/xianbao01.hou/rtpolicy_experiments/_tmp/20261021_drawer640_video_compile_vs_jpeg_bs256_rankcache_v3/results.md)。

## 5. 在当前代码中复用

三视角、16维状态/动作的真机640配置已整理为独立入口，运行前指定转换后的 **JPEG blob** 目录和任务：

```bash
conda activate RTPolicy
torchrun --standalone --nproc_per_node=8 main.py \
  --config-path=configs/rtpolicy_train \
  --config-name=dit_fm_dinov3cnnb_3v_lance_real640_bs256 \
  train_dataset.root=/efs/path/to/lance_jpeg_blob \
  train_dataset.task=drag_the_drawer_0901_renamed train_dataset.demo=224
```

该配置启用 `image_io_backend=mmap`、CPU解码、workers64、CPU预取2、单批CUDA预取和 `vision_backbone_compile=true`。原有通用配置的编译默认关闭；通过 `vision_backbone_compile=false` 可做同配置基线。机器人维度/相机/权重路径不同，应先调整配置。

正式目录只保留训练功能；固定输入、Profiler、单测和测速脚本统一放在 [EFS实验归档](/data/efs/datasets/xianbao01.hou/rtpolicy_archives/20261016_071211/README.md)。要复现诊断，请使用归档中的诊断版代码快照，保持相同数据/模型/每卡batch，先正常加载，再关闭加载重复同一真实GPU batch；计时前后核对输入哈希，Profiler另跑，不将其开销混入速度。

正式代码入口：`rtpolicy/datasets/lance_images.py`（mmap）、`rtpolicy/common/cuda_prefetch.py`（传输重叠）、`vision_compile.py`（编译）、`training_resume.py`（续训）；训练开关集中在 `rtpolicy/trainer/train.py`。

后续优化应优先验证计算融合、AdamW/EMA批量更新、CPU发射和同步，仍然一次只改一项。Roofline 的约束是 `P ≤ min(P_peak, HBM带宽 × FLOPs/HBM字节)`；SM busy高不等于有效算力用满。本节点NCU报 `ERR_NVGPUCTRPERM`，没有真实HBM流量，尚不能宣布达到硬件Roofline；独立profiler中19.8%的kernel间隙也不能直接当作可提速比例。

原始证据：`/data/efs/datasets/xianbao01.hou/rtpolicy_archives/20261016_071211/RTPolicy-backbone-compile/.train_monitor_input_ceiling_real640_20261016_055853`，含 `RESULTS.json`、每rank原始时序/输入哈希、SM曲线。完整过程、历史命令及迁移说明见EFS归档README。
