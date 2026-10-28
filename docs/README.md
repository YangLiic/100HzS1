# VFA 运行说明

使用 `python -m pip install -e .` 安装，同时提供 `rtpolicy` 和 `vfa` 两个 Python 包。
`src/rtpolicy/` 保留 RTPolicy 基线，`src/vfa/` 是独立的 VFA 开发副本；安装后使用 `import rtpolicy` 或 `import vfa`，无需在导入路径中添加 `src`。
移动目录或切换布局后，请在使用的 Python 环境中重新执行开发模式安装。
本分支仅包含运行代码和配置；数据集、预训练权重、checkpoint、RobotWin2 资产需单独准备。

训练性能优化的定位思路、实测结果和640配置见[性能优化笔记](performance_optimization.md)。

## 基线与 VFA 入口

RTPolicy 基线使用 `configs/rtpolicy_train/`；VFA 使用 `configs/vfa_train/`。
VFA 的策略类为 `vfa.models.VFADitFlowMatching`，数据集和训练器同样从 `vfa` 包加载。
VFA 配置将训练输出写入 `checkpoints/vfa/`，与基线输出分开。

先检查训练配置，再补充实际数据路径运行：

```bash
python main.py --config-path=configs/vfa_train \
  --config-name=dit_fm_dinov3cnnb_3v_lance_real640_bs256 --cfg job
```

VFA ONNX 导出入口为 `python -m vfa.trainer.onnx`，对应配置位于 `configs/vfa_onnx/`。
通用 ONNX 推理脚本 `scripts/infer_onnx.py` 可供两者使用；`scripts/infer_vfa.py` 提供 VFA 的历史缓冲示例。

RoboTwin2 评测时将策略名设为 `VFA`，加载 `policy/VFA/` 适配器：

```bash
bash scripts/eval/eval_robotwin2.0_parallel.sh \
  VFA beat_block_hammer agilex_config /absolute/path/to/checkpoint 100 2 42
```

使用 `VFA_NFE` 覆盖 VFA 的积分次数；`VFA_JPEG_ROUNDTRIP` 控制该适配器的 JPEG 输入对齐。
基线继续使用 `RTPolicy` 策略名及其原有环境变量。

## 数据准备

LeRobot 数据目录应包含 `meta/` 和 `data/`，图像可嵌入 parquet 或存放在 `videos/`。
单任务配置通过 `train_dataset.path=<任务父目录>` 和 `train_dataset.task=<任务名>`
定位数据；`train_dataset.demo=100` 选择前 100 个 episodes。

RobotWin2 原始数据转换示例：

```bash
bash scripts/dataset/robotwin2lerobot.sh /path/to/robotwin2/data /path/to/lerobot beat_block_hammer
```

其他数据准备工具：`tools/preprocess_drobotics.py`、`tools/merge_lerobot_videos.py`、
`tools/add_images_to_parquet.py`。先用各工具的 `--help` 查看输入格式。

## LeRobot v3 转 Lance

安装 Lance 依赖：`python -m pip install -e '.[lance]'`。
已有集群环境可继续使用 `--no-deps`，前提是已安装 lancedb、pylance、pyarrow 和 PyAV。
转换入口在本仓库，不需要安装 `lerobot_lancedb` 或切换 LeRobot 版本：

```bash
# MP4-backed v3 数据：保留原始视频，不额外重编码
python tools/lerobot_v30_to_lance.py \
  --root /path/to/lerobot_v3 --output /path/to/task_lance_video \
  --mode video --staging-dir /local/scratch

# v3 的 parquet 内嵌图像或 MP4：转为逐帧 JPEG blob，支持 EFS 按需读取
python tools/lerobot_v30_to_lance.py \
  --root /path/to/lerobot_v3 --output /path/to/task_lance_frames \
  --mode jpeg --jpeg-storage blob --num-workers 8 --staging-dir /local/scratch
```

| `--mode`       | 输入的相机格式                                  | Lance 产物                                                       |
| -------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| `video`        | `dtype=video`，原始 MP4                         | `<name>.lance` 数值帧表 + `<name>_videos.lance` 原始 MP4 blob 表 |
| `jpeg`（默认） | `dtype=image`（内嵌或图片路径）或 `dtype=video` | `<name>.lance` 数值 + 逐帧 JPEG 表                               |

两种方式都输出 Lance 和完整 `meta/` 侧车；`jpeg` 指逐帧 JPEG 图像存储方式。
JPEG 默认 quality=100、subsampling=0，可通过 `--jpeg-quality` 和
`--jpeg-subsampling` 修改；视频模式保留原始 MP4 字节。
纯图像源使用 `jpeg`，`video` 模式要求源数据已包含 MP4 相机。
`--jpeg-storage binary|blob` 仅用于 `jpeg` 模式，默认 `binary` 保留原布局；
EFS 多卡训练推荐 `blob`，相机列保存相同 JPEG 字节，读取器自动识别。
已有普通 binary JPEG 表可以单次重打包，不解码、不重新编码、不改源目录：

```bash
python tools/repack_lance_jpeg.py \
  --root /efs/task_lance_frames --output /efs/task_lance_jpeg_blob \
  --staging-dir /local/scratch
```

此离线转换需要能容纳完整产物的暂存空间；训练时直接读取 EFS 输出目录。

脚本按 parquet batch 流式读取数值和图像，视频文件按相机、chunk、file 去重。
多 episode 共享视频和不同相机使用不同文件索引均按 v3 metadata 处理。
`--repo-id` 或 `--table-name` 可指定表名，默认取源目录名。
`--dry-run` 只核对 metadata 和文件路径；已有输出需显式 `--overwrite`，
并且在新数据转换和帧数检查完成后才替换旧输出。源数据目录始终保留。

所有 Lance 写入先在 staging 完成，再复制到输出位置，支持 EFS 目标。
`--staging-dir` 指向能够执行原子 rename、且能容纳完整产物的本地盘；
未指定时使用 `TMPDIR` 或 `/tmp`。大数据集需预留完整产物所需空间。
v2 数据需要先升级到 v3，本脚本对其他版本会给出明确错误。

## Lance 后端

默认后端是 LeRobot。已转换的 Lance 数据使用 v3.0 `meta/` 侧车和 `.lance` 表，
在训练命令中追加：

```bash
train_dataset.backend=lance train_dataset.dataset_metadata=null \
train_dataset.root=/absolute/path/to/task_lance
```

两个后端共用 `rtpolicy.datasets.builder.build_dataset`。
Lance 自动识别逐帧表和 video 双表。也可用 `train_dataset.backend=lance_video`
显式要求视频格式。两种格式在初始化时缓存数值列，批次只读取所需图像观测。
video 按 metadata 的 `from_timestamp` 定位共享 MP4 中的帧；每个读取进程有独立
解码器 LRU，默认保留最多 16 个视频文件，可用
`+train_dataset.video_decoder_cache_size=8` 调整。

视频默认 `decode_device=cpu`，使用 PyAV；`decode_device=cuda:0` 或 `auto`
选择 NVDEC 硬件解码。安装 GPU 视频所需的 PyAV 版本：

```bash
python -m pip install -e '.[lance,lance-video]'
# 已有 RTPolicy 环境且依赖已齐全时，editable 安装仍可使用 --no-deps。
```

训练时追加（`auto` 使用当前训练 rank 的 CUDA 设备）：

```bash
train_dataset.backend=lance train_dataset.dataset_metadata=null \
train_dataset.root=/path/to/task_lance_video \
+train_dataset.decode_device=auto
```

GPU video 使用 PyAV >=17.1 的 NVDEC 原始帧和 DLPack，在 GPU 上按视频的
BT.601/BT.709/BT.2020 色彩矩阵及 full/studio range 转换 RGB；图像数据不经过 CPU。
硬件解码失败会报错，不会静默改用软件解码。源 codec/profile 必须受 GPU 的 NVDEC
支持；例如普通 H.264 4:2:0 可用，CPU 测试的 lossless H.264 4:4:4 需使用 CPU 路径。
GPU RGB 与 FFmpeg CPU RGB 可能有少量量化差异，并不承诺像素逐位一致。

GPU 解码时训练和验证加载器自动使用 `num_workers=0`、`pin_memory=false`。
CPU 路径统一读取顶层 `dataloader_num_workers`，沿用总数除以训练进程数的规则；
例如总数64、8个训练进程，每个训练/验证加载器均为8个worker。

CPU worker 路径可用下面两项进行管线对照，默认保持原来的同步传输和 PyTorch 预取深度2：

```bash
+dataloader_non_blocking=true \
+dataloader_prefetch_factor=4
```

`non_blocking` 使用 Accelerate 的非阻塞设备传输，配合 CPU 路径的 pinned memory；
这不等于已实现独立 CUDA stream 上的拷贝/计算重叠。
`prefetch_factor` 是每个 worker 的在途批次上限，训练和验证加载器共用此设置；
每个加载器的在途任务上限为实际 `num_workers × prefetch_factor`，修改worker配置后随之变化。
worker 为0时不传该参数。预取深度2改为4会增加解码后批次的内存上限，需要按图像尺寸和batch估算。
这两项都不会预载整个数据集，是否提速需在目标节点实训验证。
`+dataloader_cuda_prefetch=true` 是另一项默认关闭的管线选项：CPU 解码/pinned memory 路径
只提前一批，在独立 CUDA stream 上传输，当前 stream 等待本批就绪后即可计算，并继续传输下一批。
它要求 CUDA 训练及 `gradient_accumulation_steps=1`；GPU 解码路径不使用此选项。
该路径保留当前训练批和一个未来批的设备张量，按 stream 记录生命周期，保持样本顺序。
额外设备内存随单批大小增长，不随数据集总量增长；正常 CPU worker 预取仍有独立上限。

EFS 视频随机读取可显式调整压缩字节预读大小：
`+train_dataset.video_read_buffer_size=65536`（64 KiB；默认 1 MiB）。
短 GOP 视频的单帧包远小于 1 MiB，大缓冲会读取很多未使用的数据；具体训练收益需要实测。
`video_decoder_cache_size` 是打开的解码器数量，应结合相机数和活跃视频文件数设置。
逐 episode 小 MP4 会反复打开解码器，GPU NVDEC 的会话初始化尤其明显；
原生 v3 多 episode 共用一个 MP4 的布局仍由 metadata 的时间偏移处理。

挂载目录上的 Lance blob v1 可启用 `+train_dataset.video_io_backend=file`：
只读取 blob 的 `(position, size)` 元数据，按字节范围从原 Lance 文件读取 MP4。
每个 worker 复用有上限的物理文件句柄池，容量与 `video_decoder_cache_size` 相同；
`pread` 使不同视频的游标互不影响。视频内容不预载、不复制到本地，采样顺序不变。
默认 `lance` 仍使用 Lance BlobFile；`auto` 在支持的挂载布局上选 file，否则回到 Lance。
file 只支持 blob v1、每 fragment 一个数据文件且无外部 base 的布局；显式选择不兼容布局会报错。

JPEG blob 列默认 `image_io_backend=auto`，支持的挂载布局自动使用原生 `pread`。
初始化只读取所选相机的 blob 描述符（位置、大小、行地址），批次按偏移读 JPEG 字节；
每个 worker 最多复用16个物理文件句柄，图像载荷不整库预载，也不复制到本地。
spawn worker 只接收元数据，按需打开自己的句柄，无需重新扫描描述符。
可用 `+train_dataset.image_io_backend=lance` 对照 Lance blob API，或用 `file` 要求原生路径。
`file` 要求所有所选相机为挂载目录上的 blob v1、每 fragment 单文件且无外部 base；
不支持的布局会明确报错，`auto` 则回到 Lance API。旧 binary 列仍走原读取路径，需先重打包才能使用此优化。
描述符和既有数值列缓存仍随行数增长；本次103681帧、3相机描述符共约5.8 MB。
JPEG blob 读取要求 `image_cache_size_bytes=0`。

挂载的不可变 blob-v1 单文件 fragment 还可显式选择 `+train_dataset.image_io_backend=mmap`，
按需读取 JPEG 范围，使用 `MADV_RANDOM` / `MADV_DONTNEED` 管理本进程映射页。
本轮 EFS 长训已验证它能绕开观察到的周期性 `preadv` 停顿；使用边界和复测步骤见[性能优化笔记](performance_optimization.md)。

本次 32 episodes / 3 相机共有 96 个独立 MP4，CPU 直接读取 EFS 时，
只将 `+train_dataset.video_decoder_cache_size=96` 就恢复到本地盘训练速度。
这不是通用的固定值：按活跃视频文件数和解码器内存占用选择容量，
文件数量更多或多 rank 并发时应重新验证。

`video_cache_size_bytes` / `image_cache_size_bytes` 是历史诊断选项，
用于排除存储等待；它们预载完整压缩视频/图像列，不作为训练提速推荐。
大数据训练保持两项为 0（默认关闭）。
初始化会扫描完整的所选相机列，预算不足会明确报错；只选择部分 episodes 不会缩小预载范围。
JPEG RAM 缓存要求 `dataloader_num_workers=0`，训练与验证在主进程共享一份。
video 支持 spawn workers，但每个 worker 有一份压缩字节；
总内存应按实际数据大小乘以 `workers + 1`，多 rank 时再乘 rank 数计算，另留模型和预取空间。
缓存初始化时间不计入稳态训练 it/s，应一并考虑总运行时间。
worker数只通过顶层 `dataloader_num_workers=<总数>` 设置；已移除dataset层的重复覆盖项。
逐帧格式继续支持原来的 JPEG/NVJPEG CPU/GPU 解码选项。

可选 `+train_dataset.video_backend=torchcodec` 使用 TorchCodec 的批量 CPU 解码，
需要额外安装与 PyTorch 匹配的 TorchCodec。GPU 默认后端为 `nvdec`；没有使用
TorchCodec 0.5 的 CUDA RGB 转换，因为它对部分未标注色彩空间的 NV12 视频有范围偏差。
NVDEC 会在相邻查询相隔超过 0.25 秒时重新 seek，减少随机 batch 的中间帧过读，
可通过 `+train_dataset.video_seek_gap_s=<秒数>` 调整以匹配源视频的 GOP；CPU PyAV 默认
为 1 秒。GPU 解码仍需读取前置关键帧及参考帧，不能据此假定速度与独立 JPEG 帧相同。
多 GPU 并发访问共享存储可能受到随机 I/O 限制；进行吞吐对照时，将两种格式的数据
放在同一台机器的本地盘上。

历史测速、回归脚本及原始结果已移至 [EFS实验归档](/data/efs/datasets/xianbao01.hou/rtpolicy_archives/20261016_071211/README.md)；正式训练配置和优化方法见[性能优化笔记](performance_optimization.md)。

## 权重与旧 checkpoint

`pretrained/` 存放外部模型权重，已被 `.gitignore` 排除；Git 不记录空目录，因此新克隆的仓库通常没有此目录。
当前视觉分支使用 **DINOv3 ConvNeXt** 预训练权重；多视角融合、视觉压缩和 Flow Matching 动作头由操作数据训练。
这与已经训练好的任务策略 checkpoint 是两类文件。

训练时设置 `model.obs_encoder.emvis_config.backbone_path` 为 DINOv3 权重目录。
CNN-base / CNN-large / CNN-small 的配置分别使用
`pretrained/dinov3-cnn-base`、`pretrained/dinov3-convnext-large-pretrain-lvd1689m`、
`pretrained/dinov3-cnn-small`；以所选 YAML 的实际字段为准。

相对路径以仓库根目录为工作目录，例如 Base 配置对应 `<仓库>/pretrained/dinov3-cnn-base/`。
可以自行创建目录并放入权重，或者通过上述配置字段指定外部权重目录的绝对路径，无需复制大文件到仓库。
目录应采用 Transformers `from_pretrained()` 格式，包含 `config.json`、`model.safetensors`（或受支持的分片权重），
以及 `preprocessor_config.json` 等图像处理配置。Base、Large、Small 的权重必须与所选视觉配置匹配。
模型来源见 [DINOv3 官方仓库](https://github.com/facebookresearch/dinov3)。

VFA 的读取路径会在内存中将 RTPolicy / 旧 VODPPlus 的已知类引用映射到 `vfa`，不修改原 checkpoint。
若要永久更新一份复制出来的 checkpoint 元数据，先检查：

```bash
python tools/rename_vfa_ckpt_pkg_map.py /path/to/copied/checkpoint --dry-run
```

确认后去掉 `--dry-run`。默认只更新配置元数据，不重写权重；当前 VFA 副本与基线结构一致，后续结构改动需另行检查权重兼容性。

旧 VODPPlus checkpoint 若仍使用旧策略类名，可运行：

```bash
python tools/rename_ckpt_pkg_map.py --root /path/to/checkpoints
```

该工具修改 `config.json` 中的类名映射，保留权重文件。

## RobotWin2 外部资产

仿真运行目录为 `simulation/robotwin2.0`。资产下载入口：

```bash
cd simulation/robotwin2.0
bash script/_download_assets.sh
```

已有资产时，可将 `objects`、`background_texture` 和 `embodiments` 放入此目录的
`assets/` 下，然后运行 `python script/update_embodiment_config_path.py` 更新 cuRobo 路径。
URDF、网格、纹理及下载压缩包均由 Git 忽略。

## 输出

训练输出由 `output_dir` 指定。新实验使用独立目录，避免覆盖已有模型。
仿真视频及结果写入 `simulation/robotwin2.0/eval_result/`；ONNX/TensorRT 产物可写入
`outputs/`。这些目录、W&B 日志及 Python 缓存均不会提交。
