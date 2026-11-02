# B0 LayerNorm visual residual MLP ablations

Two explicit presets `b0_ln_fm4` and `b0_stage4_ln_fm4` replace only the
visual residual MLP's pre-RMSNorm with standard affine LayerNorm, epsilon 1e-6.
The branch is `x + Linear2048to1024(GELU(Linear1024to2048(LayerNorm(x))))`.
The raw skip, Linear128to1024, token flattening position and DiT implementation
remain unchanged. LayerNorm adds 1024 affine bias parameters.

Both presets explicitly retain the original five fusion LayerNorms (eps1e-5).
Before DiT each visual path has six LayerNorms and zero RMSNorms. DiT's existing
RMSNorms stay unchanged. No stage4: fuse80/view then two original ResBlocks to6.
Shared stage4: one frozen stage reused twice plus one trainable128to64 bridge
compress80to20to6 before fusion, no post-fusion ResBlocks. Both have no pooling,
18 visual tokens, hidden/state/action/t width1024, 8 layers and16 heads.

Controls: beat_block_hammer, three views,50 demos,seed42,200epochs/8771 updates,
batch128,BF16,GR00T q01/q99 clipping,tau_token,standard FM1000 buckets and4 steps.
Main-only closed-loop100 identical expert-verified seeds,eager/KV-on/Graph-off,
RGB unchanged. Compare against prior GR00T RMS residual runs17/100 (no stage4)
and55/100 (shared stage4), both with the original LayerNorm fusion. The recent
all-five-fusion-RMS run51/100 is a separate control and is not inherited here.
Each arm uses one ordinary end-to-end training run; no distillation or stages.
