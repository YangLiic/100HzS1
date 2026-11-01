"""Aligned compact visual/state/action tokens and nonlinear visual projection."""
from importlib import import_module
import json
from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

policy_module = import_module("100hzs1.models.policy.flow_matching.dit_fm_policy")
vision_module = import_module("100hzs1.models.policy.flow_matching.emvis.embodied_vision")
ROOT = Path(__file__).resolve().parents[1]
RGB_KEYS = ["left_cam", "right_cam", "head_cam"]


def config(variant="b0", width=256, injection="tau_token"):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    suffix = "_adarms" if injection == "tau_AdaRMS" else ""
    name = (f"backbones/efficientvit_{variant}/{injection}/"
            f"100hzs1_beat_block_hammer_efficientvit_{variant}_prefusion_2x3_meanflow{suffix}_compact_3v_8gpu_L4")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs/100hzs1_train")):
        cfg = compose(config_name=name, overrides=[f"model.hidden_size={width}"])
        OmegaConf.resolve(cfg)
        return OmegaConf.to_container(cfg)


def make_policy(variant="b0", width=256, flow="meanflow", injection="tau_token"):
    cfg = config(variant, width, injection)["model"]
    cfg["obs_encoder"]["emvis_config"]["load_backbone_pretrain"] = False
    cfg.update(flow_type=flow, time_injection=injection)
    policy = policy_module.S1DitFlowMatching(policy_module.S1DitFlowMatchingConfig(**cfg)).float()
    policy.model.dtype = policy.model.t_embedder.dtype = torch.float32
    if policy.model.r_embedder is not None:
        policy.model.r_embedder.dtype = torch.float32
    nn.init.normal_(policy.model.final_layer.ffn_final.fc2.weight, std=.02)
    policy.normalizer.fit({"state": torch.randn(32, 14), "action": torch.randn(32, 14)})
    return policy


def observation():
    obs = {k: torch.rand(1, 1, 3, 240, 320) for k in RGB_KEYS}
    obs["state"] = torch.rand(1, 1, 14)
    return obs


def linear_shapes(module):
    return [(m.in_features, m.out_features) for m in module.modules() if isinstance(m, nn.Linear)]


class CompactPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_recipes_align_all_widths_and_isolate_outputs(self):
        outputs = set()
        for variant in ("b0", "b2"):
            for width in (128, 256, 512):
                with self.subTest(variant=variant, width=width):
                    cfg = config(variant, width)
                    model = cfg["model"]
                    visual = model["obs_encoder"]["emvis_config"]
                    self.assertEqual(model["num_heads"], width // 64)
                    self.assertEqual(model["hidden_size"], width)
                    self.assertEqual(visual["model_adapter_config"]["dim_out"], width)
                    self.assertEqual(visual["output_projection_type"], "mlp2x_gelu")
                    self.assertEqual(model["scene_adaptor"], "identity")
                    self.assertEqual(model["state_adaptor"], "mlp3x_gelu")
                    self.assertEqual(model["depth"], 8)
                    self.assertEqual(cfg["horizon"], 24)
                    self.assertEqual(visual["model_adapter_config"]["shape_out"], [2, 3])
                    self.assertTrue(cfg["train_dataset"]["bgr_to_rgb"])
                    self.assertIsNone(cfg["resume_from_checkpoint"])
                    self.assertIn(f"_d{width}_", cfg["output_dir"])
                    outputs.add(cfg["output_dir"])
        self.assertEqual(len(outputs), 6)

    def test_b0_b2_actual_visual_state_time_and_kv_shapes(self):
        for variant, native in (("b0", 128), ("b2", 384)):
            with self.subTest(variant=variant):
                policy = make_policy(variant).eval()
                scene = policy.obs_encoder.scene_encoder
                self.assertEqual(linear_shapes(scene.output_projection), [(native, 256), (256, 256)])
                self.assertIsInstance(scene.output_projection[1], nn.GELU)
                self.assertIsInstance(policy.scene_adaptor, nn.Identity)
                self.assertEqual(linear_shapes(policy.state_adaptor), [(14, 256), (256, 256), (256, 256)])
                self.assertEqual(sum(isinstance(m, nn.GELU) for m in policy.state_adaptor), 2)
                self.assertEqual(policy.model.scene_cond_pos_embed.shape, (1, 18, 256))
                self.assertEqual(policy.model.x_pos_embed.shape, (1, 25, 256))
                with torch.no_grad():
                    features = policy.encode_image_conditions(observation(), 1)
                    state = policy.state_adaptor(torch.randn(1, 24, 14))
                    self.assertEqual(features.shape, (1, 18, 256))
                    self.assertEqual(state.shape, (1, 24, 256))
                    self.assertEqual(policy.model.t_embedder(torch.ones(1)).shape, (1, 256))
                    self.assertEqual(policy.model.r_embedder(torch.zeros(1)).shape, (1, 256))
                    cache = policy.model.prepare_kv_cache(features)
                    self.assertEqual(len(cache), 8)
                    for block, (k, v) in zip(policy.model.blocks, cache):
                        self.assertEqual(block.cross_attn.kv.out_features, 512)
                        self.assertEqual(k.shape, (1, 4, 18, 64))
                        self.assertEqual(v.shape, k.shape)
                    cached = policy.model(state, torch.ones(1), features, kv_cache=cache, r=torch.zeros(1))
                    plain = policy.model(state, torch.ones(1), features, r=torch.zeros(1))
                    self.assertEqual(cached.shape, (1, 24, 14))
                    torch.testing.assert_close(cached, plain, rtol=1e-5, atol=1e-6)

    def test_fm_meanflow_and_adarms_two_optimizer_steps_keep_state_mlp_gradients(self):
        for flow, injection in (("fm", "tau_token"), ("meanflow", "tau_token"),
                                ("meanflow", "tau_AdaRMS")):
            with self.subTest(flow=flow, injection=injection):
                policy = make_policy(flow=flow, injection=injection).train()
                optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
                batch = {"obs": observation(), "action": {"action": torch.rand(1, 24, 14)}}
                # Training datasets provide state trajectories over the horizon;
                # inference observations above intentionally contain one state only.
                batch["obs"]["state"] = torch.rand(1, 24, 14)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    result = policy.compute_loss(batch)
                    loss = result["loss"] if isinstance(result, dict) else result
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    checked = [m.weight for m in policy.state_adaptor if isinstance(m, nn.Linear)]
                    checked += [policy.obs_encoder.scene_encoder.output_projection[i].weight for i in (0, 2)]
                    checked.append(policy.model.blocks[0].cross_attn.kv.weight)
                    checked.append(policy.model.t_embedder.mlp[0].weight)
                    if policy.model.r_embedder is not None:
                        checked.append(policy.model.r_embedder.mlp[0].weight)
                    if injection == "tau_AdaRMS":
                        checked.append(policy.model.action_time_fusion[0].weight)
                        checked.append(policy.model.final_modulation[1].weight)
                        checked += [m[1].weight for block in policy.model.blocks for m in block.modulations]
                    for parameter in checked:
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        self.assertGreater(float(parameter.grad.abs().sum()), 0)
                    self.assertTrue(all(p.grad is None for p in policy.obs_encoder.scene_encoder.backbone.parameters()))
                    optimizer.step()

    def test_strict_json_reload_and_cached_sampling(self):
        for injection in ("tau_token", "tau_AdaRMS"):
            with self.subTest(injection=injection):
                policy = make_policy(injection=injection).eval()
                obs = observation()
                noise = torch.randn(1, 24, 14)
                with torch.no_grad():
                    expected = policy.predict_action(obs, noise_trajectory=noise, use_kv_cache=True, use_cuda_graph=False)
                    plain = policy.predict_action(obs, noise_trajectory=noise, use_kv_cache=False, use_cuda_graph=False)
                self.assertEqual(expected["action"].shape, (1, 23, 14))
                torch.testing.assert_close(expected["action"], plain["action"], rtol=1e-5, atol=1e-6)
                restored = policy_module.S1DitFlowMatching(
                    policy_module.S1DitFlowMatchingConfig(**json.loads(policy.config.to_json_string()))).float()
                restored.load_state_dict(policy.state_dict(), strict=True)
                restored.model.dtype = restored.model.t_embedder.dtype = restored.model.r_embedder.dtype = torch.float32
                restored.eval()
                self.assertEqual(restored.model.time_injection, injection)
                self.assertEqual(restored.obs_encoder.scene_encoder.output_projection_type, "mlp2x_gelu")
                with torch.no_grad():
                    actual = restored.predict_action(obs, noise_trajectory=noise, use_kv_cache=True, use_cuda_graph=False)
                torch.testing.assert_close(actual["action"], expected["action"], rtol=0, atol=0)

    def test_alternative_widths_reach_cache_without_hidden_2048(self):
        for width in (128, 512):
            with self.subTest(width=width):
                policy = make_policy(width=width).eval()
                self.assertEqual(linear_shapes(policy.state_adaptor), [(14, width), (width, width), (width, width)])
                with torch.no_grad():
                    c = policy.encode_image_conditions(observation(), 1)
                    self.assertEqual(c.shape, (1, 18, width))
                    for k, v in policy.model.prepare_kv_cache(c):
                        self.assertEqual(k.shape, (1, width // 64, 18, 64))
                        self.assertEqual(v.shape, k.shape)

    def test_both_time_injection_paths_and_recipes_use_compact_width(self):
        outputs = set()
        for variant in ("b0", "b2"):
            for injection in ("tau_token", "tau_AdaRMS"):
                for width in (128, 256, 512):
                    cfg = config(variant, width, injection)
                    self.assertEqual(cfg["model"]["time_injection"], injection)
                    self.assertEqual(cfg["model"]["hidden_size"], width)
                    self.assertEqual(cfg["model"]["meanflow"]["time_embed_scale"], 1.)
                    outputs.add(cfg["output_dir"])
        self.assertEqual(len(outputs), 12)
        for injection in ("tau_token", "tau_AdaRMS"):
            with self.subTest(injection=injection):
                policy = make_policy(injection=injection).eval()
                head = policy.model
                self.assertEqual(linear_shapes(head.t_embedder.mlp), [(256, 256), (256, 256)])
                self.assertEqual(linear_shapes(head.r_embedder.mlp), [(256, 256), (256, 256)])
                length = 25 if injection == "tau_token" else 24
                self.assertEqual(head.x_pos_embed.shape, (1, length, 256))
                self.assertEqual(head.scene_cond_pos_embed.shape, (1, 18, 256))
                first_inputs, fusion_inputs = [], []
                hook = head.blocks[0].register_forward_pre_hook(
                    lambda module, args: first_inputs.append(tuple(args[0].shape)))
                fusion_hook = None
                if injection == "tau_AdaRMS":
                    self.assertEqual(linear_shapes(head.action_time_fusion), [(512, 256), (256, 256)])
                    self.assertEqual(linear_shapes(head.final_modulation), [(256, 512)])
                    for block in head.blocks:
                        self.assertEqual(len(block.modulations), 3)
                        for modulation in block.modulations:
                            self.assertEqual(linear_shapes(modulation), [(256, 768)])
                    fusion_hook = head.action_time_fusion.register_forward_pre_hook(
                        lambda module, args: fusion_inputs.append(tuple(args[0].shape)))
                try:
                    with torch.no_grad():
                        result = policy.predict_action(observation(), use_kv_cache=True, use_cuda_graph=False)
                finally:
                    hook.remove()
                    if fusion_hook is not None:
                        fusion_hook.remove()
                self.assertEqual(result["action"].shape, (1, 23, 14))
                self.assertEqual(first_inputs, [(1, length, 256)])
                self.assertEqual(fusion_inputs, [(1, 23, 512)] if injection == "tau_AdaRMS" else [])

    def test_identity_and_projection_options_fail_on_incompatible_settings(self):
        cls = policy_module.S1DitFlowMatching
        with self.assertRaisesRegex(ValueError, "matching widths"):
            cls.build_condition_adapter(None, "identity", 128, 256)
        self.assertIsInstance(cls.build_condition_adapter(None, "identity", 256, 256), nn.Identity)
        with self.assertRaisesRegex(ValueError, "output_projection_type"):
            vision_module.EmVisRM(output_projection_type="unknown")
        with self.assertRaisesRegex(ValueError, "requires compress_before_fusion"):
            vision_module.EmVisRM(output_projection_type="mlp2x_gelu")


if __name__ == "__main__":
    unittest.main()
