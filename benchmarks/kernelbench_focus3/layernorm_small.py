import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, normalized_shape):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        return self.layer_norm(x)


batch_size = 256
dim = 4096


def get_inputs():
    return [torch.rand(batch_size, dim)]


def get_init_inputs():
    return [dim]
