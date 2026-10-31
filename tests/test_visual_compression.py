"""Pre-fusion spatial compression, checkpoint and policy contracts."""
from importlib import import_module
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

vision = import_module("100hzs1.models.policy.flow_matching.emvis.embodied_vision")
policy_module = import_module("100hzs1.models.policy.flow_matching.dit_fm_policy")
ROOT = Path(__file__).resolve().parents[1]
RGB_KEYS = ["left_cam", "right_cam", "head_cam"]


def scene_config(variant="b0", before=True):
    version = "r224" if variant == "b0" else "r256"
    return dict(
        backbone_target="efficientvit", efficientvit_fusion_mode="native",
        backbone_config={"model_name": f"efficientvit_{variant}.{version}_in1k"},
        backbone_trainable=False, load_backbone_pretrain=False,
        processor_config=dict(do_resize=False, do_rescale=False, do_normalize=True,
                              image_mean=[.485, .456, .406], image_std=[.229, .224, .225]),
        dim_2d=None, only_2d=True, only_first_view=False,
        compress_before_fusion=before,
        mv_fuser_config=dict(depth=1, num_heads=16, mlp_ratio=2., num_views=3),
        model_adapter_config=dict(target="ResNetAdapter", shape_out=[2, 3] if before else [2, 2],
                                  dim_out=1024, conv_num=2),
    )


def recipe(variant):
    name = (f"backbones/efficientvit_{variant}/tau_token/"
            f"100hzs1_beat_block_hammer_efficientvit_{variant}_prefusion_2x3_meanflow_3v_8gpu_L4")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs/100hzs1_train")):
        cfg = compose(config_name=name)
        OmegaConf.resolve(cfg)
        return OmegaConf.to_container(cfg)


def small_policy():
    cfg = recipe("b0")["model"]
    cfg.update(hidden_size=64, depth=2, num_heads=4, horizon=4, n_action_steps=3)
    cfg["obs_encoder"]["emvis_config"]["load_backbone_pretrain"] = False
    model = policy_module.S1DitFlowMatching(policy_module.S1DitFlowMatchingConfig(**cfg)).float()
    model.model.dtype = model.model.t_embedder.dtype = model.model.r_embedder.dtype = torch.float32
    # Nonzero output weights make upstream visual gradients observable immediately.
    nn.init.normal_(model.model.final_layer.ffn_final.fc2.weight, std=.02)
    model.normalizer.fit({"state": torch.randn(16, 14), "action": torch.randn(16, 14)})
    return model


class VisualCompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_b0_b2_b3_fuse_six_tokens_at_native_width_without_pooling(self):
        for variant, width in (("b0", 128), ("b2", 384), ("b3", 512)):
            with self.subTest(variant=variant):
                model = vision.VisualEncoder(emvis_config=scene_config(variant), rgb_key=RGB_KEYS).eval()
                scene = model.scene_encoder
                batch = 2 if variant == "b0" else 1
                seen = []
                def record(module, args):
                    seen.append(tuple(args[0].shape))
                hook = scene.mv_fuser.global_blocks[0].register_forward_pre_hook(record)
                try:
                    with torch.no_grad(), patch.object(scene.model_adapter.patch_merging, "forward",
                            side_effect=AssertionError("adaptive pooling must not run before fusion")):
                        out = model({k: torch.rand(batch, 3, 240, 320) for k in RGB_KEYS})
                finally:
                    hook.remove()
                self.assertEqual(seen, [(batch, 18, width)])
                self.assertEqual(out.shape, (batch, 18, 1024))
                self.assertEqual(model.output_shape_meta(), torch.Size([3, 2, 3, 1024]))
                self.assertTrue(torch.isfinite(out).all())
                self.assertEqual(scene.output_projection[0].in_features, width)

    def test_gradients_reach_compression_fusion_projection_but_not_frozen_backbone(self):
        scene = vision.EmVisRM(**scene_config()).train()
        bn = next(m for m in scene.backbone.modules() if isinstance(m, nn.BatchNorm2d))
        before = bn.running_mean.clone()
        out = scene(torch.rand(1, 3, 3, 240, 320))
        out.square().mean().backward()
        for parameter in (
                scene.model_adapter.downsample_blocks[0].mainstream.conv1.weight,
                scene.model_adapter.downsample_blocks[1].mainstream.conv1.weight,
                scene.mv_fuser.global_blocks[0].self_attn.in_proj_weight,
                scene.output_projection[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in scene.backbone.parameters()))
        self.assertFalse(bn.training)
        torch.testing.assert_close(bn.running_mean, before, rtol=0, atol=0)

    def test_single_view_and_external_tokens_follow_same_path(self):
        scene = vision.EmVisRM(**scene_config()).eval()
        images = torch.rand(2, 1, 3, 240, 320)
        with torch.no_grad():
            expected = scene(images)
            processed = scene.img_processing(images.reshape(-1, 3, 240, 320))
            tokens = scene.backbone(processed.reshape(2, 1, 3, 256, 320))
            actual = scene(image_tokens=tokens["image_tokens"], image_pos=tokens["image_tokens_pos"])
        self.assertEqual(actual.shape, (2, 1, 6, 1024))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_omitted_flag_retains_legacy_state_keys_and_fusion_first(self):
        cfg = scene_config(before=False)
        cfg.pop("compress_before_fusion")
        scene = vision.EmVisRM(**cfg).eval()
        seen = []
        hook = scene.mv_fuser.global_blocks[0].register_forward_pre_hook(
            lambda module, args: seen.append(tuple(args[0].shape)))
        with torch.no_grad():
            out = scene(torch.rand(1, 3, 3, 240, 320))
        hook.remove()
        self.assertFalse(scene.compress_before_fusion)
        self.assertEqual(seen, [(1, 240, 128)])
        self.assertEqual(out.shape, (1, 3, 4, 1024))
        self.assertIn("model_adapter.patch_merging.proj.weight", scene.state_dict())
        self.assertFalse(any(k.startswith("output_projection.") for k in scene.state_dict()))

    def test_json_roundtrip_strict_reload_and_no_duplicated_projection(self):
        cfg = scene_config()
        original = vision.VisualEncoder(emvis_config=cfg, rgb_key=RGB_KEYS).eval()
        restored = vision.VisualEncoder(emvis_config=json.loads(json.dumps(cfg)), rgb_key=RGB_KEYS).eval()
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertTrue(cfg["compress_before_fusion"])
        keys = list(original.scene_encoder.state_dict())
        self.assertIn("output_projection.0.weight", keys)
        self.assertNotIn("model_adapter.patch_merging.proj.weight", keys)
        # Moving an existing projection must not add trainable parameters.
        baseline = vision.VisualEncoder(emvis_config=scene_config(before=False), rgb_key=RGB_KEYS)
        self.assertEqual(sum(p.numel() for p in original.parameters()),
                         sum(p.numel() for p in baseline.parameters()))
        images = {k: torch.rand(1, 3, 240, 320) for k in RGB_KEYS}
        with torch.no_grad():
            torch.testing.assert_close(restored(images), original(images), rtol=0, atol=0)

    def test_invalid_modes_and_grid_fail_explicitly(self):
        for change, message in (
                ({"compress_before_fusion": "false"}, "must be a bool"),
                ({"efficientvit_fusion_mode": "projected"}, "native EfficientViT"),
                ({"backbone_target": "dinov3cnn"}, "native EfficientViT"),
                ({"model_adapter_config": None}, "requires model_adapter_config"),
                ({"model_adapter_config": {"shape_out": [0, 3]}}, "positive 2D shape_out"),
                ({"model_adapter_config": dict(shape_out=[2, 3], dim_out=1024,
                                               stage_blocks=[1], dim_ratio=[2])}, "preserve")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                vision.EmVisRM(**dict(scene_config(), **change))
        scene = vision.EmVisRM(**scene_config()).eval()
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "produced grid"):
            scene(torch.rand(1, 3, 3, 320, 320))

    def test_new_recipes_compose_and_leave_baseline_unchanged(self):
        for variant in ("b0", "b2"):
            cfg = recipe(variant)
            scene = cfg["model"]["obs_encoder"]["emvis_config"]
            self.assertTrue(scene["compress_before_fusion"])
            self.assertEqual(scene["model_adapter_config"]["shape_out"], [2, 3])
            self.assertTrue(scene["backbone_config"]["model_name"].startswith(f"efficientvit_{variant}."))
            self.assertEqual(cfg["model"]["flow_type"], "meanflow")
            self.assertEqual(cfg["inference"]["num_inference_timesteps"], 1)
            self.assertTrue(cfg["train_dataset"]["bgr_to_rgb"])
            self.assertIsNone(cfg["resume_from_checkpoint"])
            self.assertIn("prefusion_2x3", cfg["output_dir"])
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs/100hzs1_train")):
            baseline = compose(config_name="100hzs1_beat_block_hammer_efficientvit_b2_native_meanflow_3v_8gpu_L4")
            self.assertFalse(baseline.model.obs_encoder.emvis_config.get("compress_before_fusion", False))
            self.assertEqual(list(baseline.model.obs_encoder.emvis_config.model_adapter_config.shape_out), [2, 2])
            baseline_output = baseline.output_dir
        self.assertNotEqual(baseline_output, recipe("b2")["output_dir"])

    def test_policy_loss_sampling_cache_and_strict_reload_use_18_conditions(self):
        model = small_policy().train()
        self.assertEqual(model.model.scene_cond_len, 18)
        self.assertEqual(model.model.scene_cond_pos_embed.shape, (1, 18, 64))
        obs = {k: torch.rand(1, 1, 3, 240, 320) for k in RGB_KEYS}
        obs["state"] = torch.rand(1, 1, 14)
        losses = model.compute_loss({"obs": obs, "action": {"action": torch.rand(1, 4, 14)}})
        loss = losses["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.obs_encoder.scene_encoder.output_projection[0].weight.grad.abs().sum()), 0)
        model.eval()
        noise = torch.randn(1, 4, 14)
        with torch.no_grad():
            cached = model.predict_action(obs, noise_trajectory=noise, use_kv_cache=True, use_cuda_graph=False)
            plain = model.predict_action(obs, noise_trajectory=noise, use_kv_cache=False, use_cuda_graph=False)
        self.assertEqual(cached["action"].shape, (1, 3, 14))
        torch.testing.assert_close(cached["action"], plain["action"], rtol=1e-5, atol=1e-6)
        restored = policy_module.S1DitFlowMatching(
            policy_module.S1DitFlowMatchingConfig(**json.loads(model.config.to_json_string()))).float()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(restored.obs_encoder.scene_encoder.compress_before_fusion)
        self.assertEqual(restored.model.scene_cond_len, 18)
        restored.model.dtype = restored.model.t_embedder.dtype = restored.model.r_embedder.dtype = torch.float32
        restored.eval()
        with torch.no_grad():
            reloaded = restored.predict_action(obs, noise_trajectory=noise, use_kv_cache=True, use_cuda_graph=False)
        torch.testing.assert_close(reloaded["action"], cached["action"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
