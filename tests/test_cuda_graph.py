"""Inference-only CUDA Graph dispatch, lifetime, fallback and output ownership."""
from importlib import import_module as _import_module
from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import torch
from torch import nn

S1DitFlowMatching = _import_module('100hzs1.models.policy.flow_matching.dit_fm_policy').S1DitFlowMatching
EmVisRM = _import_module('100hzs1.models.policy.flow_matching.emvis.embodied_vision').EmVisRM
VisualEncoder = _import_module('100hzs1.models.policy.flow_matching.emvis.embodied_vision').VisualEncoder
BackboneCudaGraph = _import_module('100hzs1.models.policy.flow_matching.emvis.inference_cuda_graph').BackboneCudaGraph


class TinyBackbone(nn.Module):
    patch_size_log = 1

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, images):
        tokens = images * self.scale
        return {'image_tokens': tokens, 'image_tokens_pos': None}


def tiny_scene():
    scene = EmVisRM.__new__(EmVisRM)
    nn.Module.__init__(scene)
    scene.backbone = TinyBackbone()
    scene.backbone_projection = nn.Identity()
    scene.img_processing = lambda x: x
    scene.token_forward = lambda tokens: tokens['image_tokens']
    scene._backbone_cuda_graph = None
    scene.register_load_state_dict_post_hook(scene._clear_cuda_graph_after_load)
    return scene


class CudaGraphDispatchTests(unittest.TestCase):
    def test_graph_is_optional_and_disabled_by_default(self):
        self.assertIs(inspect.signature(S1DitFlowMatching.predict_action)
                      .parameters['use_cuda_graph'].default, None)
        options = _import_module('100hzs1.common.inference').load_inference_options()
        self.assertFalse(options['use_cuda_graph'])
        for function in (S1DitFlowMatching.encode_image_conditions,
                         VisualEncoder.forward, EmVisRM.forward):
            self.assertIs(inspect.signature(function)
                          .parameters['use_cuda_graph'].default, False)

    def test_cpu_falls_back_without_allocating_runtime_cache(self):
        scene = tiny_scene().eval()
        images = torch.rand(1, 3, 3, 8, 8)
        with torch.no_grad(), patch.object(BackboneCudaGraph, '_capture',
                                           side_effect=AssertionError('CPU capture')):
            expected = scene.image_forward(images)
            actual = scene.image_forward(images, use_cuda_graph=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNone(scene._backbone_cuda_graph)

    def test_graph_request_preserves_gradients_and_training(self):
        scene = tiny_scene().train()
        images = torch.rand(1, 3, 3, 8, 8, requires_grad=True)
        with patch.object(BackboneCudaGraph, '_capture', side_effect=AssertionError('capture')):
            scene.image_forward(images, use_cuda_graph=True).sum().backward()
        torch.testing.assert_close(images.grad, torch.full_like(images, 2))
        self.assertGreater(scene.backbone.scale.grad.item(), 0)
        self.assertIsNone(scene._backbone_cuda_graph)
        # Grad-enabled eval also keeps eager/autograd.
        scene.eval()
        self.assertTrue(scene.image_forward(images, use_cuda_graph=True).requires_grad)

    def test_runtime_cache_is_not_saved_and_is_invalidated(self):
        scene = tiny_scene().eval()
        before = list(scene.state_dict())
        scene._backbone_cuda_graph = object()
        self.assertEqual(before, list(scene.state_dict()))
        restored = deepcopy(scene)
        self.assertIsNone(restored._backbone_cuda_graph)
        scene.train()
        self.assertIsNone(scene._backbone_cuda_graph)
        scene._backbone_cuda_graph = object()
        scene.to(dtype=torch.float64)
        self.assertIsNone(scene._backbone_cuda_graph)
        scene._backbone_cuda_graph = object()
        scene.load_state_dict(scene.state_dict(), strict=True)
        self.assertIsNone(scene._backbone_cuda_graph)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class CudaGraphReplayTests(unittest.TestCase):
    def test_changed_inputs_shape_precision_and_owned_outputs(self):
        module = TinyBackbone().cuda().eval()
        runner = BackboneCudaGraph(module)
        for inference in (False, True):
            context = torch.inference_mode() if inference else torch.no_grad()
            with context:
                for bf16 in (False, True):
                    with torch.autocast('cuda', enabled=bf16, dtype=torch.bfloat16):
                        for shape in ((1, 3, 3, 8, 8), (2, 3, 3, 8, 8)):
                            x = torch.rand(*shape, device='cuda')
                            first = runner(x)['image_tokens']
                            saved = first.clone()
                            second = runner(x + 1)['image_tokens']
                            torch.testing.assert_close(first, saved, rtol=0, atol=0)
                            torch.testing.assert_close(second, module(x + 1)['image_tokens'],
                                                       rtol=0, atol=0)
                            self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(runner.captures, 8)
        self.assertEqual(runner.replays, 16)

    def test_capture_failure_falls_back_once_for_signature(self):
        module = TinyBackbone().cuda().eval()
        runner = BackboneCudaGraph(module)
        x = torch.rand(1, 3, 3, 8, 8, device='cuda')
        with torch.no_grad(), patch.object(runner, '_capture', side_effect=RuntimeError('test failure')) as capture:
            with self.assertLogs('100hzs1.models.policy.flow_matching.emvis.inference_cuda_graph', level='WARNING'):
                first = runner(x)
            second = runner(x + 1)
            self.assertEqual(capture.call_count, 1)
            torch.testing.assert_close(first['image_tokens'], module(x)['image_tokens'])
            torch.testing.assert_close(second['image_tokens'], module(x + 1)['image_tokens'])

    def test_post_begin_error_is_not_silently_swallowed(self):
        module = TinyBackbone().cuda().eval()
        runner = BackboneCudaGraph(module)
        x = torch.rand(1, 3, 3, 8, 8, device='cuda')
        def fail_after_begin(*args):
            runner._capture_started = True
            raise RuntimeError('simulated capture error')
        with torch.no_grad(), patch.object(runner, '_capture', side_effect=fail_after_begin):
            with self.assertRaisesRegex(RuntimeError, 'Restart inference'):
                runner(x)

    def test_real_host_sync_is_rejected_before_capture_and_rng_stays_usable(self):
        class HostSync(TinyBackbone):
            def forward(self, images):
                value = images.sum().item()
                return {'image_tokens': images * self.scale + value}
        module = HostSync().cuda().eval()
        runner = BackboneCudaGraph(module)
        x = torch.rand(1, 3, 3, 8, 8, device='cuda')
        with torch.no_grad():
            with self.assertLogs('100hzs1.models.policy.flow_matching.emvis.inference_cuda_graph', level='WARNING'):
                actual = runner(x)['image_tokens']
            torch.testing.assert_close(actual, module(x)['image_tokens'], rtol=0, atol=0)
            actual = runner(x + 1)['image_tokens']
            torch.testing.assert_close(actual, module(x + 1)['image_tokens'], rtol=0, atol=0)
            self.assertEqual(runner.captures, 0)
            self.assertTrue(torch.isfinite(torch.randn(4, device="cuda")).all())

    def test_scene_explicit_off_bypass_and_train_clears_capture(self):
        scene = tiny_scene().cuda().eval()
        images = torch.rand(1, 3, 3, 8, 8, device='cuda')
        with torch.no_grad():
            scene.image_forward(images, use_cuda_graph=True)
            runner = scene._backbone_cuda_graph
            self.assertEqual(runner.replays, 1)
            expected = scene.image_forward(images, use_cuda_graph=False)
            self.assertEqual(runner.replays, 1)
            actual = scene.image_forward(images + 1, use_cuda_graph=True)
            torch.testing.assert_close(actual, expected + 2, rtol=1e-6, atol=1e-6)
            self.assertEqual(runner.replays, 2)
        scene.train()
        self.assertIsNone(scene._backbone_cuda_graph)


if __name__ == '__main__':
    unittest.main()
