import numpy as np
import torch
from torch import nn
from env.game import Bigtwo
from utils.checkpoint import checkpoint
import numpy as np


class Bigtwo156(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)
        self.dense1 = nn.Linear(52 + 52 + 52, 256)
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


class Bigtwo156Numpy:
    def __init__(self, state_dict):
        self.weights1 = np.array(state_dict["dense1.weight"])
        self.bias1 = np.array(state_dict["dense1.bias"])
        self.weights2 = np.array(state_dict["dense2.weight"])
        self.bias2 = np.array(state_dict["dense2.bias"])
        self.weights3 = np.array(state_dict["dense3.weight"])
        self.bias3 = np.array(state_dict["dense3.bias"])

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        x = np.dot(x, self.weights1) + self.bias1
        x = self.relu(x)
        x = np.dot(x, self.weights2) + self.bias2
        x = self.relu(x)
        x = np.dot(x, self.weights3) + self.bias3
        return x

    def act(self, x):
        x = self.forward(x)
        return np.argmax(x)


class Agent156:
    def __init__(self, device):
        self.histories = []
        self.rewards = []
        self.device = torch.device(device)
        self.model = Bigtwo156(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, game):
        if len(self.histories) < len(self.rewards) + 1:
            self.histories.append([])
        obs = self.observe(game)
        with torch.no_grad():
            output = self.model(obs["x_batch"])
        action_index = torch.argmax(output, dim=0)[0]
        self.histories[-1].append(obs["x_batch"][action_index])
        action = game.players[game.player_to_act].legal_actions[action_index]
        game.step(action)

    def learn(self):
        losses = []
        for i, history in enumerate(self.histories):
            self.optimizer.zero_grad()
            x_batch = torch.stack(history)
            output = self.model(x_batch.float())
            y_batch = torch.ones(x_batch.shape[0], 1).to(self.device) * self.rewards[i]
            loss = torch.nn.functional.mse_loss(output, y_batch)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach())
        self.histories = []
        self.rewards = []
        return torch.mean(torch.tensor(losses)).cpu().item()

    def observe(self, game: Bigtwo):
        players = game.players
        player_to_act = game.player_to_act
        player = players[player_to_act]
        legal_actions = player.legal_actions
        legal_actions = torch.tensor(np.array([a.code for a in legal_actions])).to(
            self.device
        )
        holding = torch.tensor(player.holding).to(self.device)
        other_indices = [
            (i + player_to_act) % 4
            for i in range(4)
            if (i + player_to_act) % 4 != player_to_act
        ]
        others_holding = [players[i].holding for i in other_indices]
        others_holding = np.bitwise_or.reduce(others_holding, axis=0)
        others_holding = torch.tensor(others_holding).to(self.device)
        x = torch.cat([holding, others_holding], dim=0)
        x_batch = x.repeat(len(legal_actions), 1)
        x_batch = torch.cat([x_batch, legal_actions], dim=1).float()
        return dict(
            x_batch=x_batch,
        )

    def save(self, id="default"):
        checkpoint(self.model, self.optimizer, f"bigtwo156-{id}")
