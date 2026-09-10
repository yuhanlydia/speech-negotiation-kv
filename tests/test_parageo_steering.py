import torch
from torch import nn

from speech_negotiation_kv.parageo_steering import ScheduledFusedQKVSteerer


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_key_value = nn.Identity()

    def forward(self, x):
        return self.query_key_value(x)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_scheduled_steering_advances_once_per_forward_and_saturates():
    model = Tiny()
    directions = torch.tensor([[1., 2., 3., 4.], [10., 20., 30., 40.]])
    x = torch.zeros(1, 1, 3)
    with ScheduledFusedQKVSteerer(
        model, layer_indices=[0, 1], directions=directions,
        num_attention_heads=1, kv_channels=1, multi_query_group_num=1,
    ) as steerer:
        first = model(x)
        second = model(x)
        third = model(x)
        assert steerer.step_index == 2
    assert torch.allclose(first, torch.tensor([[[0., 4., 6.]]]))
    assert torch.allclose(second, torch.tensor([[[0., 40., 60.]]]))
    assert torch.allclose(third, second)
