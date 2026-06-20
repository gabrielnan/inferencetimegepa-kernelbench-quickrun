import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        super(Model, self).__init__()
        layers = []
        current_input_size = input_size
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        layers.append(nn.Linear(current_input_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


batch_size = 64
input_size = 4096
layer_sizes = [4096, 4096]
output_size = 2048


def get_inputs():
    return [torch.rand(batch_size, input_size)]


def get_init_inputs():
    return [input_size, layer_sizes, output_size]
