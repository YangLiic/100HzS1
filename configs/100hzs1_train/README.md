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

## Visual token compression

EfficientViT can compress each view to its natural 2x3 grid before multiview
fusion with `compress_before_fusion: true`. This path uses two existing ResBlocks,
skips adaptive pooling, and projects native channels to 1024 after fusion.
B2 recipes under `backbones/efficientvit_b2/tau_token/` contain `prefusion_2x3`
in their filenames and use independent checkpoint output paths. B3 also has an
observation profile. Existing recipes retain fusion before 2x2 compression.
See [the dimensions, configuration and validation](../../docs/visual_token_compression.md).

## Compact policy width

Recipes with `meanflow_compact` in their filenames align visual, state/action,
time and DiT features at 256 dimensions. The compact visual projector aligns the backbone output to that width. The state/action encoder keeps all three Linear layers and
both GELU activations, using `14 -> 256 -> 256 -> 256` instead of 2048-wide
layers. The extra scene MLP is Identity because its input is already aligned.
The DiT retains eight layers and uses four 64-dimensional heads.

`model.hidden_size=128` or `model.hidden_size=512` selects alternative widths;
visual output size, head count and the checkpoint output path follow that value.
Existing 2048-wide recipes remain unchanged. This is a new training architecture,
so these recipes use independent paths and do not resume old policy checkpoints.
See [the full computation and validation](../../docs/visual_token_compression.md).

Both `tau_token/` and `tau_AdaRMS/` retain B2 compact recipes. AdaRMS filenames
use `meanflow_adarms_compact` and independent output paths. The shared hidden
width also controls time embeddings, action-time fusion and every AdaRMS
scale/shift/gate projection; no 2048-wide time path remains in compact models.

## B0 controlled FM experiments

Only three B0 FM4 experiment entrypoints are maintained. Both ablations inherit
the original 2048-wide baseline, with the same training/data/inference settings.

| Recipe suffix | Visual compression | Scene output / state / action / time / DiT width |
|---|---|---:|
| `native_fm_d2048_baseline_3v_8gpu_L4` | fuse80/view -> two ResBlocks -> pool2x2 | 2048 |
| `stage4repeat_2x2_fm_d2048_3v_8gpu_L4` | shared stage4x2 -> pool2x2 -> fuse4/view | 2048 |
| `resmlp_fm_d1024_3v_8gpu_L4` | baseline compression, residual RMS scene MLP | 1024 |

All are under `backbones/efficientvit_b0/tau_token/` with prefix
`100hzs1_beat_block_hammer_efficientvit_b0_` and suffix `.yaml`.
Every recipe retains Linear128->1024 and 12 final visual condition tokens.
Stage4 pooling is before multiview fusion, adds zero parameters, and has no
post-fusion ResBlocks. The residual recipe keeps the baseline ResBlocks/pool.

Standard FM / tau_token / four inference steps, 50demo / seed42 / 200epochs /
batch128 / BF16 / three views / minmax / KV-on / Graph-off remain common.
Output paths are separated by recipe and timestamp; no old checkpoint is resumed.
DiT implementation is unchanged: eight layers and fixed D->D->GELU->D FFNs.
Only the existing hidden-width/head-count settings differ for the 1024 ablation.

The old direct-width1024 and six-token-stage4 YAMLs are archived outside the
Hydra search path at `docs/archived_configs/beat_b0_fm4_20261101/`.
Old checkpoints/results/frozen snapshots are untouched.
See [the three full computation paths](../../docs/beat_b0_fm4_three_configs_20261101.md).
