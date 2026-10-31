# Training configurations

Training recipes are organized under `backbones/<backbone>/<time_injection>/`.
Choose `tau_token` or `tau_AdaRMS`, then select the YAML for the task, flow
objective, camera setup and batch size. The shared `dataset_config`,
`observation_config`, `policy_config`, `robot_config` and `inference` groups
remain at this level because multiple backbones use them.

For example:

```bash
scripts/train/train.sh configs/100hzs1_train/backbones/efficientvit_b2/tau_AdaRMS/100hzs1_beat_block_hammer_efficientvit_b2_native_meanflow_adarms_3v_8gpu_L4.yaml
```

The original YAML paths remain as compatibility aliases, so existing launch
commands and external scripts continue to work. Edit the corresponding recipe
under `backbones/` rather than its alias. Moving a recipe does not change its
training settings or checkpoint output path.

## State/action normalization

Set the top-level scalar `norm` in a recipe, or override it on the training command:

| norm | Behavior |
| --- | --- |
| `minmax` | Original per-dimension min/max mapping; default for omitted fields |
| `minmax_expanded` | Expand the min/max interval 1.25x; training extrema map to approximately -0.8 and 0.8 |
| `pi` | q01/q99 mapping with epsilon 1e-6; no clipping |
| `groot` | q01/q99 mapping; near-constant dimensions map to zero; clip normalized state/targets and final predictions to [-1,1] |

For example, append `norm=pi` to `scripts/train/train.sh <recipe.yaml>`.
The choice is applied to absolute joint `state` and supervised `action` fields.
Both LeRobot and Lance compute the new modes from selected training frames;
validation uses the policy's training normalizer. Delta/endpose recipes continue
to require `norm=minmax`, retaining their existing specialized processing.

The model's config.json stores `norm`; scale, offset, quantile statistics and any
clipping/inverse parameters are saved in the normalizer state. Old configs and
weights without `norm` load as `minmax` without rebuilding their statistics or
adding required state tensors. Inference uses checkpoint parameters.
Resuming optimizer state, or keeping a checkpoint normalizer, requires a matching
normalization mode. A fresh training run is required for a controlled mode comparison.

Ready-to-run beat_block_hammer recipes keep DINOv3 ConvNeXt-Base, FM,
tau_token and 200 epochs fixed:

```bash
scripts/train/train.sh configs/100hzs1_train/normalization/beat_dinov3_base_tau_token/pi.yaml
```

Choose `minmax.yaml`, `minmax_expanded.yaml`, `pi.yaml`, or `groot.yaml` in
that directory. Each mode has a separate checkpoint output directory. Override
`output_dir` when repeating an experiment so existing results are preserved.
