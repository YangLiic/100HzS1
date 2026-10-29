"""Compare installed FM against the original policy source saved in each checkpoint."""
import argparse
import gc
import importlib.util
from importlib import import_module
import json
from pathlib import Path
import sys
import torch
from safetensors.torch import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(1)
    load_policy = import_module('100hzs1.models.base_runner').load_policy
    for checkpoint in args.checkpoint:
        config = json.loads((checkpoint/'config.json').read_text())
        assert config.get('flow_type', 'fm') == 'fm'
        name = '100hzs1.models.policy.flow_matching._legacy_checkpoint_reference'
        spec = importlib.util.spec_from_file_location(name, checkpoint/'dit_fm_policy.py')
        legacy = importlib.util.module_from_spec(spec);sys.modules[name] = legacy;spec.loader.exec_module(legacy)
        original = legacy.S1DitFlowMatching(legacy.S1DitFlowMatchingConfig.from_dict(config)).cuda().eval()
        current = load_policy(checkpoint, device='cuda:0').eval()
        torch.manual_seed(900)
        obs = {key: torch.rand((1, 1, *meta['shape']), device='cuda:0')
               for key, meta in config['shape_meta']['obs'].items()}
        for weights in ('main', 'ema'):
            path = checkpoint/('model.safetensors' if weights == 'main' else 'ema/model.safetensors')
            load_model(original, str(path), strict=True, device='cuda:0')
            load_model(current, str(path), strict=True, device='cuda:0')
            assert set(original.state_dict()) == set(current.state_dict())
            for graph in (False, True):
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    torch.manual_seed(42);a = original.predict_action(obs, use_cuda_graph=graph)['action']
                    torch.manual_seed(42);b = current.predict_action(obs, use_cuda_graph=graph)['action']
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                print(json.dumps(dict(checkpoint=str(checkpoint), weights=weights, cuda_graph=graph,
                    strict_reload=True, bitwise_equal=True, max_abs_error=float((a-b).abs().max()),
                    shape=list(b.shape), nfe=current.num_inference_timesteps)), flush=True)
        del original, current, legacy
        gc.collect();torch.cuda.empty_cache()

if __name__ == '__main__': main()
