"""EMA must carry train-time normalization state, including saved precision."""
import copy
from importlib import import_module
import unittest
import torch
from torch import nn

EMAModel = import_module("100hzs1.common.ema_model").EMAModel

class TestEMABatchNorm(unittest.TestCase):
    def test_updates_bn_buffers_and_averages_other_parameters(self):
        live = nn.Sequential(nn.Linear(4,4,bias=False),nn.BatchNorm1d(4))
        average = copy.deepcopy(live)
        ema = EMAModel(average)
        for i in range(3):
            live.train()
            live(torch.arange(32,dtype=torch.float32).reshape(8,4)+i*20)
            with torch.no_grad():live[0].weight.add_(0.2)
            before = average[0].weight.clone()
            ema.step(live)
            for key, value in live[1].state_dict().items():
                torch.testing.assert_close(average[1].state_dict()[key],value,rtol=0,atol=0)
            expected = before*ema.decay+live[0].weight*(1-ema.decay)
            torch.testing.assert_close(average[0].weight,expected)
        self.assertFalse(average.training)
        self.assertEqual(average[1].num_batches_tracked.item(),3)

    def test_bfloat16_ema_and_frozen_backbone_buffers(self):
        live = nn.Sequential(nn.BatchNorm2d(4),nn.BatchNorm2d(4))
        live[0].requires_grad_(False).eval()
        live[1].train()
        frozen={k:v.clone() for k,v in live[0].state_dict().items()}
        average=copy.deepcopy(live).bfloat16()
        ema=EMAModel(average)
        for i in range(3):
            live(torch.arange(256,dtype=torch.float32).reshape(4,4,4,4)+i*10)
            ema.step(live)
        for key,value in live.state_dict().items():
            actual=average.state_dict()[key]
            torch.testing.assert_close(actual,value.to(actual.dtype),rtol=0,atol=0)
        for key,value in live[0].state_dict().items():
            torch.testing.assert_close(value,frozen[key],rtol=0,atol=0)
        self.assertEqual(average[0].num_batches_tracked.item(),0)
        self.assertEqual(average[1].num_batches_tracked.item(),3)

if __name__=="__main__":unittest.main()
