# Conditional FM and pure MeanFlow

Training selects `model.flow_type: fm | meanflow`. Missing fields in legacy
checkpoints mean FM. The existing package/class paths are preserved; FM creates
no additional tensors and retains its original noise-to-action time convention,
loss normalization, sampler, normalizers and state/action slicing.

Use `configs/100hzs1_train/100hzs1_beat_block_hammer_dinov3cnnb_meanflow_3v_8gpu_L4.yaml`
for DINOv3 ConvNeXt Base. It inherits the FM data/backbone/optimizer recipe:
50 demonstrations, 200 epochs, global batch 128, frozen visual backbone,
three RGB views, one current state, horizon24 / 23 future commanded actions.
Output is an independent directory with `meanflow` in its name.

```bash
NUM_PROCESSES=8 GPU_IDS=0,1,2,3,4,5,6,7 bash scripts/train/train.sh \
  configs/100hzs1_train/100hzs1_beat_block_hammer_dinov3cnnb_meanflow_3v_8gpu_L4.yaml
```

MeanFlow uses z_t=(1-t)*action+t*noise and v=noise-action, 0<=r<=t<=1.
It regresses u toward stopgrad(v-(t-r)*JVP(u;(v,1,0))). Observed state tokens
and inactive joints are fixed; their JVP tangents are zero. JVP includes the
raw-action adapter and continuous float time embeddings. New training YAMLs
save `meanflow.time_embed_scale: 1.0`, matching MP1's continuous time inputs.
Missing/null scale preserves the v1 MeanFlow checkpoint convention, which used
`num_train_timesteps` (1000). Both training and sampling use the saved scale;
changing it on existing weights changes their meaning. FM keeps its original
bucket scaling. Image conditions are fixed with
respect to flow time. They are encoded once; their trainable adapters still
receive gradients through the separate optimization forward pass.

Only MeanFlow instantiates `model.r_embedder`; the default `tau_token` path adds
its embedding to the existing time token, preserving positional shapes. Primitive attention is used
only for the detached JVP pass and original attention flags are restored before
the parameter-gradient forward. Guidance is disabled (`guidance_scale: 1.0`;
other values explicitly rejected), and there is no Dispersive Loss. The default `equal_time_ratio: 0.0` samples interval supervision without
forcing any `t=r` samples. Set it to `0.5` to assign half of each local batch
to instantaneous-velocity supervision and half to interval supervision. Adaptive L2
matches the MP1 recipe: gamma=0.5, epsilon=0.001, with active future dimensions
used for each sample's MSE. `loss_type: mse` is also available.
Logging separates `meanflow_mse_equal_time` / `meanflow_mse_interval` and their
adaptive losses, and records velocity, JVP and correction RMS. Empty subsets
report zero; inspect `equal_time_ratio` to distinguish absent supervision.

### Time injection architecture

`model.time_injection` accepts `tau_token` (default) or `tau_AdaRMS`. Missing
checkpoint fields resolve to `tau_token`, including original FM and MeanFlow
weights. The switch changes the network architecture: it is **not** a runtime
inference option, and weights trained with one value cannot be strictly loaded
or resumed using the other. The resume and checkpoint initialization paths
reject mismatches. Existing checkpoint directories are not modified.

`tau_AdaRMS` separately embeds continuous `t` and `r` and sums them into `c`.
Only MeanFlow has `r`; standard FM keeps its existing discrete time convention.
New MeanFlow AdaRMS requires `meanflow.time_embed_scale: 1.0` (the default for
this architecture), while the old token-mode missing-scale semantics remain
unchanged. The common state adaptor encodes the observed prefix and future
actions; a full-width `Linear(2D,D) -> GELU -> Linear(D,D)` fuses `[action,c]`
for every future token. Prefix tokens skip this fusion. No time token is added;
the position embedding keeps the original state/action positions.

Each DiT block has independent full-width `D -> 3D` projections for the three
pre-norms. Each produces time-dependent scale, shift, and an unconstrained
residual gate: `x + gate * sublayer(RMSNorm(x)*(1+scale)+shift)`. Gates start at
1, scale/shift at 0, and projection weights at 0, retaining an ungated residual
path at initialization. The output RMSNorm has a separate `D -> 2D` projection
for scale/shift (no output gate). All time-dependent paths are inside the
function differentiated by MeanFlow's JVP. For 8 layers and width 2048 this
full gated variant adds about 323 million parameters to the token-mode action
head. It requires a newly trained checkpoint and its success/latency remain to
be measured; the separate PixArt-inspired shared design is not implemented.

Example B2 recipe:
`configs/100hzs1_train/100hzs1_beat_block_hammer_efficientvit_b2_native_meanflow_adarms_3v_8gpu_L4.yaml`.
It uses a separate output directory and does not start training by itself.

The initial v1 runs did not learn reliable one-step interval prediction. Their
finite outputs and strict reload checks establish execution, not convergence.
The v2 YAMLs use independent output directories and the corrected time scale;
they require small convergence validation before another formal training run.

Inference reconstructs the algorithm from checkpoint metadata. Defaults resolve
as algorithm defaults -> checkpoint inference -> selected runtime YAML -> CLI.
An existing FM checkpoint defaults to 10 NFE; the MeanFlow recipe saves 1 NFE.
Explicit runtime NFE can change the number of intervals without changing the
trained algorithm. MeanFlow performs z_r=z_t-(t-r)*u(z_t,t,r,condition), so
one step uses t=1,r=0. No JVP is needed during inference. PyTorch direct,
open-loop, RoboTwin and ONNX full-policy export share the policy sampler;
ONNX metadata includes flow_type. Visual CUDA Graphs, K/V caching and BF16
remain available. K/V reuse has little benefit at one NFE.

Legacy config `action_key` still migrates to canonical `state_key` as before.
Both old Base/B2 model and EMA weights must load strictly. The compatibility
checker compares actions against the Python source originally saved in each
checkpoint with fixed observations/noise, in eager and visual-Graph modes.
The existing FM checkpoint is a baseline, not a MeanFlow model. Cross-objective
checkpoint initialization/resume is rejected; intentional migration requires
an explicit future implementation of time/sign and new-parameter handling.

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p test_meanflow.py -v
PYTHONPATH=src .venv/bin/python tools/check_policy_checkpoint.py /absolute/checkpoint
PYTHONPATH=src .venv/bin/python tools/check_legacy_fm_compatibility.py /absolute/fm_checkpoint
```

Evaluation must use the same committed 100 expert seeds and runtime precision,
Graph/KV settings for all policies. Offline videos require BGR-to-RGB; online
SAPIEN is RGB and is not flipped. Compare both success rate and isolated L4
latency (visual, full policy with GPU inputs, and full policy with host inputs),
including warm-up and p50/p95. MeanFlow adaptive loss is not directly comparable
to the FM training loss. Record algorithm and actual NFE per checkpoint in
benchmark outputs. Use a frozen source snapshot for multi-hour training.

References: LogSSim/MP1 commit e1e9e8fc60da9137b699ad4544d34fcb0222e624,
`MP1/mp1/policy/meanpolicy.py`, and the original MeanFlow identity.
