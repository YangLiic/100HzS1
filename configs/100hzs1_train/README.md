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

## B0 visual variants

Only four B0 visual architectures are maintained, under
`backbones/efficientvit_b0/tau_token/`:

| Recipe | Before fusion | After fusion | Final width |
|---|---|---|---:|
| `b0_baseline_fm4.yaml` | original B0, 80 tokens/view | original two ResBlocks, pool2x2, Linear128->1024, original scene MLP | 2048 |
| `b0_stage4_fm4.yaml` | shared frozen stage4 twice, 6 tokens/view | pool2x2, Linear128->1024, original scene MLP | 2048 |
| `b0_rms_fm4.yaml` | original B0, 80 tokens/view | original two ResBlocks, pool2x2, Linear128->1024, RMS residual scene MLP | 1024 |
| `b0_stage4_rms_fm4.yaml` | shared frozen stage4 twice, 6 tokens/view | pool2x2, Linear128->1024, RMS residual scene MLP | 1024 |

All four use the original adaptive average pool **after fusion**, producing
four tokens/view and 12 visual tokens in total. Shared-stage4 variants replace
the original two ResBlocks with exactly two extra calls to the same frozen B0
stage4 and one shared trainable 128->64 bridge (8192 parameters). Pooling adds
no parameters. There is no independent MBConv or post-fusion stage4 branch.

`model.visual_variant` is the only architecture selector:
`baseline`, `stage4`, `rms`, or `stage4_rms`. The three ablation YAMLs inherit
baseline and change only that value. It derives the policy width, heads,
scene adaptor and ResBlock count; conflicting manual overrides are rejected.
Retired compression/width switches and their old recipes have been removed.

The residual visual MLP is `u + FC2048->1024(GELU(FC1024->2048(RMSNorm(u))))`.
DiT implementation is unchanged: eight layers and fixed D->D->GELU->D FFNs,
with 32 heads at D=2048 or 16 at D=1024. State/action/time follow D.

Standard FM / tau_token / four inference steps, 50demo / seed42 / 200epochs /
batch128 / BF16 / three views / minmax / KV-on / Graph-off remain common.
Output paths are separated by variant and timestamp; no old checkpoint is resumed.
Other normalization modes and unrelated backbone/time/flow recipes are preserved.

The four structures require the user's review before starting the newly requested
training and 100-fixed-seed evaluation of all four policies, including baseline.
See [the full paths and verification](../../docs/beat_b0_four_visual_variants_20261102.md).
