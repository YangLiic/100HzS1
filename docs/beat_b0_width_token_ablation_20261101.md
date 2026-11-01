# Beat B0: width and token-compression ablations

The previous compact experiment changed both token compression (post-fusion 2x2
to pre-fusion 2x3) and policy width (2048 to 256), including the visual projection
and scene adaptor. Its 1/100 result does not identify a causal component.
These two new arms follow the user's requested separation relative to the
original B0/native/2x2/1024/2048 architecture. Neither arm resumes the 256-wide policy.

| Arm | Compression | Visual Linear | Scene MLP | State/action/time/DiT |
|---|---|---|---|---|
| A, width only | Original fusion of 80 tokens/view, then two ResNet downsamplers and adaptive pooling to 2x2 | 128->256 | 256->512->GELU->512 | 512; 8 layers; 8 heads |
| B, compression only | Backbone stage4 repeated twice before fusion; natural 2x3, no ResNet/adaptive pooling | 128->1024 | 1024->2048->GELU->2048 | 2048; 8 layers; 32 heads |

Both scene MLPs use `mlp2x_gelu`, with one GELU between the two Linear layers.
The visual projection itself is a single Linear, with no extra visual MLP or GELU.
The state/noisy-action adaptor remains `mlp3x_gelu`: 14->D->GELU->D->GELU->D.
Both t and r embeddings use D. DiT head width remains64. KV per layer is
[B,8,12,64] in A and [B,32,18,64] in B. Physical action/state width remains14.

## Shared stage4 implementation

B0 stage4 has an entry MBConv64->128 with spatial stride2, followed by two
128-wide EfficientViT blocks. Repeating the full stage directly on128 channels
is invalid. The new encoder therefore uses one shared trainable Conv1x1
128->64 (8192 parameters, no bias), then calls the original stage4 module.
The same bridge and same stage4 parameters are reused for both extra passes:

```
256x320 RGB -> original backbone -> 128x8x10
  -> bridge128->64 -> shared stage4 -> 128x4x5
  -> same bridge  -> same stage4   -> 128x2x3
  -> flatten6/view -> multiview fusion at128 -> Linear128->1024
  -> scene MLP1024->2048->GELU->2048 ->18 conditions
```

The original ImageNet trunk, including stage4 and its BatchNorm statistics,
stays frozen in both arms. Gradients traverse its repeated operations to train
the new bridge in B. This bridge is part of the changed compression mechanism;
it is not a copied/randomly initialized stage4. No stage4 parameter is duplicated
in the optimizer or state dictionary. Default `stage4_repeats: 0` preserves old
weights and code paths. New B checkpoints serialize the bridge and repeat count.

Inputs remain240x320 and use the existing bilinear resize to256x320, without
square cropping. Preprocessing alignment remains32, not128, so the resize itself
does not change when stage4 repeats are enabled.

## Common experiment recipe

beat_block_hammer, 50 demonstrations, seed42,49 train/1 validation, absolute
joint14 actions, minmax normalization, three views, horizon24 (1 state+23 actions),
MeanFlow/tau_token, time scale1, equal-time ratio0, guidance1, BF16,
200 epochs, global batch128, AdamW lr1e-4, video shared RAM cache enabled.
Offline training converts the known BGR videos to RGB. SAPIEN evaluation remains RGB.

Both arms use final MAIN weights for closed-loop evaluation, PyTorch/eager BF16,
NFE1, KV enabled, CUDA Graph disabled, with exactly the same100 expert-verified
beat/demo_clean seeds (SHA256 fca2035b8f623d52f980843f8f578710c520997db5c7f334f557ebec5a2ec8da).
Main and EMA are saved and strictly reloaded. Do not substitute seeds, run an EMA
comparison, or attribute an effect solely to width/token count without a matched
original baseline and consideration of the changed compression mechanism.

Training entry points:

```
backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_postfusion_2x2_meanflow_d512_3v_8gpu_L4
backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_stage4repeat_2x3_meanflow_d2048_3v_8gpu_L4
```

Validation:22 tests passed across test_visual_ablation.py (5),
test_visual_backbones.py (10), and test_compact_policy.py (7). These cover actual
stage input/output grids, one shared parameter set, bridge gradients, frozen BN,
fusion input lengths, JSON strict reload, old checkpoint architecture contracts,
and both time-injection paths. GPU real-data smoke and final strict reload are
additional runtime gates in the frozen campaign, not assumed from these CPU tests.
