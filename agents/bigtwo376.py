import numpy as np
import torch
from torch import nn
from env.game import Bigtwo
from utils.checkpoint import checkpoint
import numpy as np


class Bigtwo376(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)
        self.lstm = nn.LSTM(208, 64, batch_first=True)
        self.dense1 = nn.Linear(64 + 52 + 52 + 52 * 3 + 52, 256)
        # lstm output + holding + others_holding + other played + legal_actions
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 1)
        self.to(self.device)

    def forward(self, x, z):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :]
        x = torch.cat([lstm_out, x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        return x


class Bigtwo376Numpy:
    def __init__(self, state_dict):
        self.weights1 = np.array(state_dict["dense1.weight"]).T
        self.bias1 = np.array(state_dict["dense1.bias"]).T
        self.weights2 = np.array(state_dict["dense2.weight"]).T
        self.bias2 = np.array(state_dict["dense2.bias"]).T
        self.weights3 = np.array(state_dict["dense3.weight"]).T
        self.bias3 = np.array(state_dict["dense3.bias"]).T

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


class Agent376:
    def __init__(self, device):
        self.histories = []
        self.rewards = []
        self.device = torch.device(device)
        self.model = Bigtwo376(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, game):
        if len(self.histories) < len(self.rewards) + 1:
            self.histories.append([])
        obs = self.observe(game)
        with torch.no_grad():
            output = self.model(obs["x_batch"], obs["z_batch"])
        action_index = torch.argmax(output, dim=0)[0]
        self.histories[-1].append(
            dict(x=obs["x_batch"][action_index], z=obs["z_batch"][action_index])
        )
        action = game.players[game.player_to_act].legal_actions[action_index]
        game.step(action)

    def learn(self):
        losses = []
        for i, history in enumerate(self.histories):
            self.optimizer.zero_grad()
            x_batch = torch.stack([h["x"] for h in history])
            z_batch = torch.stack([h["z"] for h in history])
            output = self.model(x_batch, z_batch)
            y_batch = torch.ones(x_batch.shape[0], 1).to(self.device) * self.rewards[i]
            loss = torch.nn.functional.mse_loss(output, y_batch)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach())
        self.histories = []
        self.rewards = []
        return torch.mean(torch.tensor(losses)).cpu().item()

    def observe(self, game: Bigtwo):
        empty = np.zeros(52, dtype=bool)
        players = game.players
        player_to_act = game.player_to_act
        player = players[player_to_act]
        legal_actions = player.legal_actions
        legal_actions = torch.tensor(np.array([a.code for a in legal_actions])).to(
            self.device
        )
        holding = torch.tensor(player.holding).to(self.device)
        history = [t[1] for t in game.traces]
        all_played = np.bitwise_or.reduce(history, axis=0) if history else empty
        others_holding = np.bitwise_not(np.bitwise_or(player.holding, all_played))
        others_holding = torch.tensor(others_holding).to(self.device)
        others_played = [history[::-1][i::4] for i in range(4)][:3]
        others_played = [
            np.bitwise_or.reduce(h, axis=0) if h else empty for h in others_played
        ]
        others_played = np.concatenate(others_played)
        others_played = torch.tensor(others_played).to(self.device)

        x = torch.cat([holding, others_holding, others_played], dim=0)
        x_batch = x.repeat(len(legal_actions), 1)
        x_batch = torch.cat([x_batch, legal_actions], dim=1).float()

        last_16_actions = np.zeros((16, 52), dtype=bool)
        for i, action in enumerate(history[::-1][:16]):
            last_16_actions[i] = action
        z = last_16_actions.reshape(-1, 208)
        z_batch = (
            torch.tensor(z)
            .to(self.device)
            .float()
            .unsqueeze(0)
            .repeat(len(legal_actions), 1, 1)
        )

        return dict(
            x_batch=x_batch,
            z_batch=z_batch,
        )

    def save(self, id="default"):
        checkpoint(self.model, self.optimizer, f"bigtwo376-{id}")
