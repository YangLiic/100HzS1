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
| `minmax` | Original per-dimension min/max mapping; retained for explicit/legacy recipes |
| `minmax_expanded` | Expand the min/max interval 1.25x; training extrema map to approximately -0.8 and 0.8 |
| `pi` | q01/q99 mapping with epsilon 1e-6; no clipping |
| `groot` | Default for new training: q01/q99 mapping; near-constant dimensions map to zero; clip normalized state/targets and final predictions to [-1,1] |

For example, append `norm=pi` to `scripts/train/train.sh <recipe.yaml>`.
The choice is applied to absolute joint `state` and supervised `action` fields.
Both LeRobot and Lance compute the new modes from selected training frames;
validation uses the policy's training normalizer. Delta/endpose recipes continue
to require `norm=minmax`, retaining their existing specialized processing.

The model's config.json stores `norm`; scale, offset, quantile statistics and any
clipping/inverse parameters are saved in the normalizer state. Old serialized checkpoints without `norm` load as `minmax` without rebuilding their statistics or
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

New recipes default to **GR00T-style normalization (`norm: groot`), no pooling
(`model.visual_pooling: false`), standard FM, tau_token and four inference steps**.
Explicit overrides remain supported. Existing run snapshots and saved checkpoints
retain their recorded settings; these new defaults do not rewrite old results.

Recipes live under `backbones/efficientvit_b0/tau_token/`:

| visual_variant / recipe suffix | Repeat stage4 before fusion | Visual projection | Scene adaptor | DiT width |
|---|---|---|---|---:|
| `baseline` | no | 128 -> 1024 | ordinary 1024 -> 2048 -> GELU -> 2048 | 2048 |
| `stage4` | yes | 128 -> 1024 | ordinary 1024 -> 2048 -> GELU -> 2048 | 2048 |
| `rms` | no | 128 -> 1024 | RMSNorm + residual 1024 -> 2048 -> GELU -> 1024 | 1024 |
| `stage4_rms` | yes | 128 -> 1024 | RMSNorm + residual 1024 -> 2048 -> GELU -> 1024 | 1024 |
| `mlp1024` | no | 128 -> 512 | ordinary 512 -> 1024 -> GELU -> 1024 | 1024 |
| `stage4_mlp1024` | yes | 128 -> 512 | ordinary 512 -> 1024 -> GELU -> 1024 | 1024 |
| `stage4_resmlp_no_norm` | yes | 128 -> 1024 | residual 1024 -> 2048 -> GELU -> 1024, explicitly no RMSNorm | 1024 |

Each named recipe is `b0_<visual_variant>_fm4.yaml`. The variant fixes width,
heads, scene adaptor and original ResBlock count; conflicting manual overrides
are rejected. Independent DiT FFN width switches remain removed.

All new B0 recipes retain the natural 2x3 grid (6 tokens/view, 18 across three
views). Without repeat, the original two trainable ResBlocks downsample after
fusion from 8x10 to 2x3. With repeat, those ResBlocks are absent: the same frozen
B0 stage4 is called two extra times before fusion, 8x10 -> 4x5 -> 2x3, using one
shared trainable 128->64 bridge (8192 parameters). Fusion is unchanged. There is
no independent MBConv or post-fusion stage4 branch.

`model.visual_pooling=true` explicitly restores post-fusion AdaptiveAvgPool2d
2x2, 4 tokens/view and 12 total; it adds zero parameters. Legacy checkpoint
metadata specifying 2x2 retains pooling when reloaded. Changing condition count
requires fresh training of the complete policy.

`rms` always means the complete visual **RMSNorm + residual MLP**:
`u + FC2048->1024(GELU(FC1024->2048(RMSNorm(u))))`.
The separately named no-norm variant is the explicitly authorized ablation
`u + FC2048->1024(GELU(FC1024->2048(u)))`; it is never called rms.

DiT remains eight layers with fixed D->D->GELU->D FFNs, head_dim=64,
32 heads for D=2048 or 16 for D=1024. State, noisy actions and time follow D.
Shared controls: 50 demos, seed42, 200 epochs, batch128, BF16, three views,
KV-on / Graph-off; standard FM uses 1000 time buckets and four inference steps.
Use independent output paths for repeated experiments; never resume a different
normalization or architecture as if it were the same experiment.
