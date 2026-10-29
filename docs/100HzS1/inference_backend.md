# 推理后端

共享配置：`src/100hzs1/config/inference.yaml`（训练配置中的 inference/default 指向同一文件）。

```yaml
backend: eager             # eager (PyTorch) | tensorrt
precision: bf16
use_cuda_graph: false
use_kv_cache: true
num_inference_timesteps: 10
tensorrt_batch_size: 1
tensorrt_engine_dir: null
```

TensorRT 可直接用于 RoboTwin 闭环；RGB→动作的模型计算由 TensorRT 执行，观测更新、动作执行保持原控制流程。
默认使用 PyTorch BF16；TensorRT 必须显式选择 fp16 或 fp32。未将 INT8 PTQ 配方作为部署默认值。
eager 支持 fp16/bf16/fp32；TensorRT 支持 fp16/fp32。CUDA Graph 独立开关，默认关闭。

```bash
pip install -e '.[tensorrt]'
bash scripts/eval/eval_robotwin.sh --checkpoint <checkpoint> --gpus 0
# 使用 TensorRT：追加 --backend tensorrt --precision fp16
# 开启 CUDA Graph：追加 --cuda-graph
```

RoboTwin `setup_env.sh` 默认仅安装 PyTorch 运行依赖；需要 TensorRT 时设置 `S1_INSTALL_TENSORRT=1`。
首次加载在控制开始前导出/构建引擎；之后复用缓存。默认缓存为运行目录的 `.cache/100hzs1/tensorrt`，可用 `tensorrt_engine_dir` 或 `S1_TENSORRT_CACHE` 指定共享目录。
缓存按权重、配置、源码、精度、batch、GPU 类型和 TensorRT/CUDA 版本隔离；跨 GPU 类型需要重建。失败会明确报错，不静默退回 eager。

统一 API：

```python
from importlib import import_module
load = import_module('100hzs1.inference').load_inference_policy
with load(checkpoint, device='cuda:0') as policy:
    result = policy.predict_action(obs, seed=42)
    actions = result['action']  # NumPy [B, n_action_steps, action_dim]
```

相机输入 `[B,T,3,H,W]`、RGB float32 `[0,1]`（也接受 uint8）；state `[B,T,D]`。两后端共用 NumPy 采样噪声，支持显式 `noise_trajectory` 做数值对照。返回值拥有独立内存。
eager 的 Graph 捕获视觉主干；TensorRT 的 Graph 捕获完整引擎执行序列。两者均支持 FM / MeanFlow 和配置的采样步数。
旧 checkpoint 保留采样步数/KV 设置，使用当前 PyTorch BF16、Graph 关闭的部署默认值；无需重训或改写权重。
训练/导出仍使用 `models.base_runner.load_policy` 的 Torch 对象；部署、开环评测和闭环控制使用 `load_inference_policy`。

验证（L4，B2 native / DINO Base 最终权重）：40 项回归通过；两模型 4/10 步 TensorRT、Graph 开关和连续控制适配器调用通过。FP16 对 eager FP32 动作相对 L2 误差 0.046%–0.179%。未运行完整仿真任务或真机。
