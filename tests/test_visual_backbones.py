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
