# B0 stage4 + RMS residual MLP: RMSNorm in visual fusion

The `b0_stage4_rms_fm4` preset explicitly sets
`model.obs_encoder.emvis_config.mv_fuser_config.norm_type: rmsnorm`.
At fusion depth 1 this replaces all five fusion LayerNorm modules: two in
the per-view encoder, two in the cross-view encoder, and the fusion output norm.
Their epsilon remains 1e-5. The existing scene RMSNorm keeps epsilon 1e-6.
Before DiT the visual path has five fusion RMSNorms plus one scene RMSNorm.

The frozen pretrained EfficientViT-B0 and its shared stage4, trainable 128-to-64
bridge, view embeddings, attention/FFN weights and residual ordering, visual
Linear 128-to-1024, scene RMSNorm residual MLP 1024-to-2048-to-1024, and DiT
implementation are unchanged. With pooling disabled, three views yield 18
visual tokens. Only the five 128-element normalization biases are removed
(640 parameters). Each new run is a single end-to-end training run.

The RMSNorm encoder overrides the explicit PreNorm forward: PyTorch's parent
TransformerEncoderLayer fused inference path hard-codes LayerNorm semantics.
No global attention fast paths are disabled. Old checkpoint metadata without
`norm_type` constructs the original LayerNorm modules and keeps strict loading;
switching a trained LayerNorm checkpoint to RMSNorm is not function-preserving.
Existing frozen campaigns retain their own implementation and configuration.

Experiment controls: beat_block_hammer/demo_clean, three cameras, 50 demos,
seed 42, 200 epochs/8771 updates, GR00T q01/q99 normalization with clipping,
no pooling, tau_token, standard FM with 1000 buckets and four inference steps,
BF16, eager inference, KV on, CUDA Graph off, closed-loop RGB unchanged.
Compare with the previous GR00T stage4_rms result of 55/100 using the same seeds.
