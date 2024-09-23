"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import numpy as np

import torch
from torch import nn


class BigtwoModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)
        self.dense1 = nn.Linear(52 + 52 + 52, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 1)
        self.to(self.device)

    def forward(self, x, return_value=False, flags=None):

        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        if return_value:
            return dict(values=x)
        else:
            if (
                flags is not None
                and flags.exp_epsilon > 0
                and np.random.rand() < flags.exp_epsilon
            ):
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x, dim=0)[0]
            return dict(action=action)


class Model:
    """
    The wrapper for the three models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """

    def __init__(self, device=0):
        if not device == "cpu":
            device = "cuda:" + str(device)

        self.models = {
            p: BigtwoModel(device).to(torch.device(device)) for p in range(4)
        }

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(x, training, flags)

    def share_memory(self):
        for model in self.models.values():
            model.share_memory()

    def eval(self):
        for model in self.models.values():
            model.eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
