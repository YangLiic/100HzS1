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

B0 dimension/token-compression experiments use standard FM and four inference
steps unless MeanFlow is explicitly requested. The old B0 MeanFlow experimental
recipes and their local training/evaluation artifacts were removed by request.
B0 pretrained weights, the native B0 profile, implementation code and other
backbones remain available.

- `backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_postfusion_2x2_fm_d1024_3v_8gpu_L4.yaml`:
  original fusion and 2x2 compression; Linear128->512, flatten3x4=12,
  scene MLP512->1024->GELU->1024; state/action/t/DiT1024 (standard FM has no r embedder).
- `backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_stage4repeat_2x3_fm_d2048_3v_8gpu_L4.yaml`:
  two additional calls of the same frozen pretrained stage4 with one shared
  trainable128->64 bridge, 6 tokens/view before fusion; Linear128->1024,
  scene MLP1024->2048->GELU->2048; state/action/t/DiT2048 (standard FM has no r embedder).
  No independent MBConv downsamplers and no adaptive2x2 pooling in this case.

Both recipes retain beat_block_hammer, three views, min-max state/action,
50 demonstrations, seed42, 200 epochs, BF16, tau_token, KV-on and Graph-off.
They initialize new policy heads and have separate output paths.
