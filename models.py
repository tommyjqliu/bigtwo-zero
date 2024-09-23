import torch
from torch import nn


class Bigtwo104(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)
        self.dense1 = nn.Linear(52 + 52, 256)
        # self holding cards + action
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 1)
        self.to(self.device)

    def forward(self, x):

        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        return x


class Bigtwo156(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)
        self.dense1 = nn.Linear(52 + 52 + 52, 256)
        # self holding cards + other holding + action
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 1)
        self.to(self.device)

    def forward(self, x):
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        return x
