"""Backbone compatibility, spatial contract, frozen BN and offline reload."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

import torch
from torch import nn

from vfa.models.policy.vfa.emvis.embodied_vision import EmVisRM, VisualEncoder
from vfa.models.policy.vfa.emvis.encoder.efficientvit_encoder import EfficientViTEncoder


def scene_config(variant="b2", dim=1024):
    return dict(
        backbone_target="efficientvit",
        backbone_config={"model_name": f"efficientvit_{variant}.r256_in1k"},
        backbone_trainable=False, load_backbone_pretrain=False,
        backbone_path="/nonexistent/pretrained",
        processor_config=dict(do_resize=False, do_rescale=False, do_normalize=True,
                              image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]),
        dim_2d=dim, only_2d=True, only_first_view=False, mlp_ratio=1.,
        mv_fuser_config=dict(depth=1, num_heads=16, mlp_ratio=2., drop_p=0.,
                             use_view_embed=True, num_views=3),
        model_adapter_config=dict(target="ResNetAdapter", shape_out=[2, 2], dim_out=1024, conv_num=2),
    )


class VisualBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_dino_original_state_keys_and_forward_are_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(
            ["git", "show", "26edd29:src/vfa/models/policy/vfa/emvis/embodied_vision.py"],
            cwd=root, text=True,
        )
        legacy = types.ModuleType("vfa.models.policy.vfa.emvis.legacy_embodied_vision")
        legacy.__package__ = "vfa.models.policy.vfa.emvis"
        exec(compile(source, "<committed_dino_baseline>", "exec"), legacy.__dict__)
        config = dict(
            dim_2d=32, backbone_target="dinov3cnn", load_backbone_pretrain=False,
            backbone_config=dict(hidden_sizes=[8, 16, 24, 32], depths=[1, 1, 1, 1]),
            processor_config=dict(do_resize=False, do_rescale=False),
            only_2d=True, mlp_ratio=1.,
            mv_fuser_config=dict(depth=1, num_heads=4, mlp_ratio=2., num_views=3, drop_p=0.),
            model_adapter_config=dict(target="ResNetAdapter", shape_out=[2, 2], dim_out=32, conv_num=2),
        )
        torch.manual_seed(42)
        original = legacy.EmVisRM(**deepcopy(config)).eval()
        torch.manual_seed(42)
        updated = EmVisRM(**deepcopy(config)).eval()
        self.assertEqual(original.state_dict().keys(), updated.state_dict().keys())
        updated.load_state_dict(original.state_dict(), strict=True)
        image = torch.rand(1, 3, 3, 240, 320)
        with torch.no_grad():
            torch.testing.assert_close(original(image), updated(image), rtol=0, atol=0)
        self.assertEqual(tuple(updated.img_processing(image[:, 0]).shape), (1, 3, 256, 320))

    def test_native_b2_b3_spatial_tokens_and_coordinates(self):
        for variant, channels in [("b2", 384), ("b3", 512)]:
            with self.subTest(variant=variant):
                model = EfficientViTEncoder(f"efficientvit_{variant}.r256_in1k")
                with torch.no_grad():
                    output = model(torch.rand(1, 3, 3, 256, 320))
                self.assertEqual(output["image_tokens"].shape, (1, 3, 80, channels))
                pos = output["image_tokens_pos"]
                self.assertEqual(pos.shape, (1, 3, 80, 2))
                self.assertEqual(pos[0, 0, 0].tolist(), [0, 0])
                self.assertEqual(pos[0, 0, -1].tolist(), [7, 9])
                self.assertFalse(any(k.startswith("head.") for k in model.model.state_dict()))

    def test_projection_gradients_and_frozen_batchnorm(self):
        model = EmVisRM(**scene_config()).train()
        bn = next(m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d))
        before = bn.running_mean.clone()
        self.assertFalse(model.backbone.training)
        self.assertFalse(bn.training)
        output = model(torch.rand(1, 3, 3, 240, 320))
        self.assertEqual(output.shape, (1, 3, 4, 1024))
        output.square().mean().backward()
        self.assertIsNotNone(model.backbone_projection.weight.grad)
        self.assertGreater(model.backbone_projection.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        torch.testing.assert_close(before, bn.running_mean, rtol=0, atol=0)

    def test_checkpoint_constructor_and_strict_reload_need_no_external_weights(self):
        config = scene_config()
        model = VisualEncoder(emvis_config=deepcopy(config), rgb_key=["left_cam", "right_cam", "head_cam"]).eval()
        serialized = json.loads(json.dumps(config))
        restored = VisualEncoder(emvis_config=serialized, rgb_key=model.rgb_key).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(model.output_shape_meta(), torch.Size([3, 2, 2, 1024]))
        images = {key: torch.rand(1, 3, 240, 320) for key in model.rgb_key}
        with torch.no_grad():
            actual, expected = restored(images), model(images)
        self.assertEqual(actual.shape, (1, 12, 1024))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_local_pretrained_loader_is_strict_for_trunk_and_ignores_classifier(self):
        from safetensors.torch import save_file
        model = EfficientViTEncoder("efficientvit_b2.r256_in1k")
        with tempfile.TemporaryDirectory(prefix="vfa_efficientvit_test_") as directory:
            path = Path(directory) / "model.safetensors"
            state = {k:v.clone() for k,v in model.model.state_dict().items()}
            state["head.fake_classifier"] = torch.zeros(1)
            save_file(state, str(path))
            restored = EfficientViTEncoder("efficientvit_b2.r256_in1k")
            restored.load_pretrained_model(path)
            for key, value in model.model.state_dict().items():
                torch.testing.assert_close(restored.model.state_dict()[key], value, rtol=0, atol=0)
            state.pop(next(k for k in state if k.endswith("conv.weight")))
            save_file(state, str(path))
            with self.assertRaises(RuntimeError):
                restored.load_pretrained_model(path)

    def test_unfrozen_backbone_can_train_and_update_batchnorm(self):
        model = EfficientViTEncoder("efficientvit_b2.r256_in1k", frozen=False).train()
        self.assertTrue(model.training)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(next(m for m in model.modules() if isinstance(m, nn.BatchNorm2d)).training)


    def test_native_fusion_and_compression_widths_gradients_and_frozen_bn(self):
        for variant, channels in (("b2", 384), ("b3", 512)):
            with self.subTest(variant=variant):
                config = scene_config(variant)
                # Native mode must derive its width even if old dim_2d is present.
                config["efficientvit_fusion_mode"] = "native"
                model = EmVisRM(**config).train()
                self.assertEqual(model.dim_2d, channels)
                self.assertIsInstance(model.backbone_projection, nn.Identity)
                self.assertFalse(any(k.startswith("backbone_projection.") for k in model.state_dict()))
                self.assertEqual(model.mv_fuser.dim, channels)
                self.assertEqual(model.model_adapter.dim_mid, [channels] * 3)
                self.assertEqual(model.dim_out, 1024)
                bn = next(m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d))
                before = bn.running_mean.clone()
                self.assertFalse(bn.training)
                output = model(torch.rand(1, 3, 3, 240, 320))
                self.assertEqual(output.shape, (1, 3, 4, 1024))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                for parameter in (model.mv_fuser.frame_blocks[0].self_attn.in_proj_weight,
                                  model.model_adapter.patch_merging.proj.weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
                torch.testing.assert_close(before, bn.running_mean, rtol=0, atol=0)

    def test_native_json_config_strict_reload_and_policy_interface(self):
        for variant in ("b2", "b3"):
            with self.subTest(variant=variant):
                config = scene_config(variant)
                config.update(efficientvit_fusion_mode="native", dim_2d=None)
                model = VisualEncoder(emvis_config=deepcopy(config),
                                      rgb_key=["left_cam", "right_cam", "head_cam"]).eval()
                # Reconstruct from the config that would be saved in a checkpoint.
                serialized = json.loads(json.dumps(config))
                serialized["load_backbone_pretrain"] = False
                restored = VisualEncoder(emvis_config=serialized, rgb_key=model.rgb_key).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                self.assertEqual(model.output_shape_meta(), torch.Size([3, 2, 2, 1024]))
                images = {key: torch.rand(1, 3, 240, 320) for key in model.rgb_key}
                with torch.no_grad():
                    expected, actual = model(images), restored(images)
                self.assertEqual(actual.shape, (1, 12, 1024))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_projected_legacy_weights_and_optimizer_resume_are_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(
            ["git", "show", "8a7a28a:src/vfa/models/policy/vfa/emvis/embodied_vision.py"],
            cwd=root, text=True,
        )
        legacy = types.ModuleType("vfa.models.policy.vfa.emvis.legacy_projected_vision")
        legacy.__package__ = "vfa.models.policy.vfa.emvis"
        exec(compile(source, "<committed_projected_baseline>", "exec"), legacy.__dict__)
        # Small trainable tower; same production B2 trunk and state-key layout.
        config = scene_config(dim=32)
        original = legacy.EmVisRM(**deepcopy(config)).train()
        optimizer = torch.optim.AdamW(original.parameters(), lr=1e-4)
        image = torch.rand(1, 3, 3, 240, 320)
        optimizer.zero_grad(set_to_none=True)
        original(image).square().mean().backward()
        optimizer.step()
        state, optimizer_state = deepcopy(original.state_dict()), deepcopy(optimizer.state_dict())
        for mode in (None, "projected"):
            with self.subTest(mode=mode):
                reload_config = deepcopy(config)
                if mode is not None:
                    reload_config["efficientvit_fusion_mode"] = mode
                updated = EmVisRM(**reload_config).train()
                self.assertEqual(list(original.state_dict()), list(updated.state_dict()))
                self.assertEqual([n for n, _ in original.named_parameters()],
                                 [n for n, _ in updated.named_parameters()])
                updated.load_state_dict(state, strict=True)
                restored_optimizer = torch.optim.AdamW(updated.parameters(), lr=1e-4)
                restored_optimizer.load_state_dict(deepcopy(optimizer_state))
                original.load_state_dict(state, strict=True)
                optimizer.load_state_dict(deepcopy(optimizer_state))
                optimizer.zero_grad(set_to_none=True)
                restored_optimizer.zero_grad(set_to_none=True)
                expected, actual = original(image), updated(image)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected.square().mean().backward()
                actual.square().mean().backward()
                optimizer.step()
                restored_optimizer.step()
                for key, value in original.state_dict().items():
                    torch.testing.assert_close(updated.state_dict()[key], value, rtol=0, atol=0)

    def test_native_configs_compose_without_changing_legacy_experiments(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
        root = Path(__file__).resolve().parents[1]
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        with initialize_config_dir(version_base=None, config_dir=str(root / "configs/vfa_train")):
            for variant in ("b2", "b3"):
                old_name = f"blocks_ranking_rgb_efficientvit_{variant}_3v_8gpu_L4"
                new_name = f"blocks_ranking_rgb_efficientvit_{variant}_native_3v_8gpu_L4"
                legacy, native = compose(config_name=old_name), compose(config_name=new_name)
                for cfg in (legacy, native):
                    OmegaConf.resolve(cfg)
                old_vision = legacy.model.obs_encoder.emvis_config
                new_vision = native.model.obs_encoder.emvis_config
                self.assertEqual(old_vision.get("efficientvit_fusion_mode", "projected"), "projected")
                self.assertEqual(old_vision.dim_2d, 1024)
                self.assertEqual(new_vision.efficientvit_fusion_mode, "native")
                self.assertIsNone(new_vision.dim_2d)
                self.assertEqual(new_vision.model_adapter_config.dim_out, 1024)
                self.assertNotEqual(legacy.output_dir, native.output_dir)
                self.assertEqual(native.base_dir, native.output_dir)
                self.assertEqual(OmegaConf.to_container(legacy.train_dataset),
                                 OmegaConf.to_container(native.train_dataset))
                # Everything outside the chosen visual width and experiment path stays identical.
                expected = OmegaConf.to_container(legacy)
                actual = OmegaConf.to_container(native)
                for cfg in (expected, actual):
                    cfg.pop("output_dir")
                    cfg.pop("base_dir")
                    vision = cfg["model"]["obs_encoder"]["emvis_config"]
                    vision.pop("dim_2d")
                    vision.pop("efficientvit_fusion_mode", None)
                self.assertEqual(actual, expected)

    def test_native_rejects_conflicting_widths_and_unknown_modes(self):
        config = scene_config()
        config["efficientvit_fusion_mode"] = "unknown"
        with self.assertRaisesRegex(ValueError, "efficientvit_fusion_mode"):
            EmVisRM(**config)
        for name in ("mv_fuser_config", "model_adapter_config"):
            with self.subTest(config=name):
                config = scene_config()
                config["efficientvit_fusion_mode"] = "native"
                config[name]["dim_in"] = 1024
                with self.assertRaisesRegex(ValueError, "native EfficientViT width 384"):
                    EmVisRM(**config)
