"""Compare the actual TensorRT output against the same checkpoint's FP32 eager output."""
import argparse,importlib,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--case',type=Path,required=True);ap.add_argument('--steps',type=int,required=True);args=ap.parse_args()
    tr=importlib.import_module('100hzs1.inference.tensorrt'); opt=importlib.import_module('100hzs1.common.inference'); eg=importlib.import_module('100hzs1.inference.eager');meta=json.loads((args.case/'bundle/metadata.json').read_text())
    overrides={'num_inference_timesteps':args.steps,'use_cuda_graph':False,'use_kv_cache':True,'tensorrt_batch_size':1}
    topt=opt.checkpoint_inference_options(args.checkpoint,overrides=dict(overrides,backend='tensorrt',precision='fp16'))
    eopt=opt.checkpoint_inference_options(args.checkpoint,overrides=dict(overrides,backend='eager',precision='fp32'))
    tr.ensure_engines=lambda checkpoint,options,hardware,device:(args.case/'bundle',meta)
    native=tr.TensorRTPolicy(args.checkpoint,'cuda:0',topt);ref=None
    try:
        rng=np.random.default_rng(4231);obs={k:rng.random((1,meta['n_image_obs_steps'],*meta['raw_image_shapes'][k]),dtype=np.float32) for k in meta['rgb_keys']};state=np.zeros((1,meta['n_obs_steps'],meta['action_dim']),np.float32);obs[meta['state_key']]=state
        if 'delta_' in meta['state_key']:obs[meta['state_key'].replace('delta_','')]=state
        noise=rng.standard_normal((1,meta['horizon'],meta['action_dim'])).astype(np.float32)
        out=native.predict_action(obs,noise_trajectory=noise)['action'];ref=eg.EagerPolicy(args.checkpoint,'cuda:0',eopt);expected=ref.predict_action(obs,noise_trajectory=noise)['action']
        delta=out.astype('float64')-expected.astype('float64');assert np.isfinite(delta).all()
        result={'steps':args.steps,'reference':'PyTorch FP32 eager, same generated RGB/state/noise','samples':int(delta.size),'max_abs':float(np.abs(delta).max()),'rmse':float(np.sqrt((delta**2).mean())),'reference_rms':float(np.sqrt((expected.astype('float64')**2).mean())),'all_finite':True}
        (args.case/f'parity_nfe{args.steps}.json').write_text(json.dumps(result,indent=2));print('PARITY_COMPLETE',json.dumps(result),flush=True)
        if result['rmse'] > 0.01 or result['max_abs'] > 0.05: raise RuntimeError('TensorRT FP16 parity exceeds RMSE 0.01 or max absolute error 0.05; inspect before reporting')
    finally:
        if ref is not None:ref.close()
        native.close()
if __name__=='__main__':main()
