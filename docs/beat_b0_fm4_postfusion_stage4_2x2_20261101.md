# B0 FM4: post-fusion shared stage4, retained 2x2 pooling

User-requested single experiment on machine2; running native2048 baseline on machine1 remains unchanged in its own frozen snapshot. Same beat_block_hammer, 50 demonstrations, seed42, 200 epochs/8771 optimizer steps, three views, minmax, BF16, standard FM/tau_token/1000 time buckets, four inference steps. Final main model gets the same 100 fixed expert-verified seeds, eager/BF16/KV-on/Graph-off/RGB-no-flip and per-episode videos.

Per view: frozen pretrained B0 -> 128x8x10 -> original multiview fusion at128 dimensions -> shared trainable1x1 bridge128->64 + original frozen stage4 ->128x4x5 -> same bridge + same stage4 ->128x2x3 -> original adaptive average pooling2x2 -> original Linear128->1024. Flatten three views to12tokens; original sceneMLP1024->2048->GELU->2048, state/action/t/DiT2048, 8layers/32heads. No r branch, independent MBConv or original two ResBlocks in this configuration.

Only the two ResBlocks are replaced. Pooling/projection implementations remain unchanged. Stage4 includes its entry MBConv and two EfficientViT blocks; one module/parameter set is used for the backbone's original call and both additional calls. One8192-parameter bridge is shared across calls/views. All pretrained weights and BatchNorm statistics remain frozen; gradients through frozen operators reach the bridge and the preceding multiview fuser. No no_grad/detach on this post-fusion path.

The opt-in `backbone_config.stage4_repeat_position: post_fusion` defers additional calls until after fusion. Default `backbone` preserves existing configs/state keys. A pool-only ResNetAdapter (`conv_num: null`, empty stage_blocks/dim_ratio) retains the original AdaptivePatchMerging and128->1024 projection. Invalid simultaneous ResBlock/post-fusion compression is rejected.

New YAML: configs/100hzs1_train/backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_poststage4_2x2_fm_d2048_3v_8gpu_L4.yaml. Unique campaign beat_b0_fm4_postfusion_stage4_2x2_20261101_v1. Freeze source/config and run CPU real-data preflight, eight-GPU two-step smoke with strict main/EMA reload, then training and strict final reload before closed-loop. Separate CPU W&B relay must show actual server progress. No push or deletion.

CPU regression tests exercise operation order80-token fusion before both repeats, natural2x3 input to unchanged2x2pool/projection, nonzero finite bridge/fuser gradients, frozen B0/BN, singleton shared parameter registration, JSON config strict reload, invalid double-compression rejection, and legacy before-fusion behavior. Launch evidence and final results are recorded under the campaign run root.

Pre-launch CPU regression: 28 tests passed across visual ablation(7), visual compression(8), compact policy and independent MBConv legacy tests(13). No GPU used on the shared development machine.

## Final audited result

Completed 8771 training steps, final loss0.0024. Finalmain/EMA strict reload, frozen B0 and actual FM4/order/pooling checks passed. Finalmain closed-loop **8/100**, shards2+3+2+1, exactly100pinnedseeds and100nonemptyvideos. Request succeeded exit0 at2026-11-01T14:03:30UTC andmachine2returnedholding. W&B6aapwos1 FINISHED8771.

Full audit: runs/beat_b0_fm4_postfusion_stage4_2x2_20261101_v1/final_audit.json. Four-way report: docs/beat_b0_fm4_four_way_results_20261101.md.

Aggregate metadata originally inherited machine1 label from a dispatcher literal. Actual sealedplan/request/supervisor/evaleventhostname prove machine2 execution. Original result.json and frozen dispatcher preserved; new result_machine2_verified.json corrects only this label and adds provenance hashes, with all episode/video/checkpoint/seed fields unchanged. The full auditor verifies this correction and recomputes8successes from100rows. No rerun/source/config/weight changes.
