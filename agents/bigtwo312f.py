import numpy as np
import torch
from torch import nn
from env.game import Bigtwo
from utils.checkpoint import checkpoint
import numpy as np


class Bigtwo312f(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)
        self.dense1 = nn.Linear(52 + 52 + 52 * 3 + 52, 256)
        # holding + others_holding + other played + legal_actions
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 128)
        self.dense4 = nn.Linear(128, 128)
        self.dense5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.to(self.device)

    def forward(self, x, training=True):
        y = self.dense1(x)
        y = torch.relu(y)
        if training:
            y = self.dropout(y)
        y = self.dense2(y)
        y = torch.relu(y)
        if training:
            y = self.dropout(y)
        residual = y
        y = self.dense3(y)
        y = torch.relu(y)
        if training:
            y = self.dropout(y)
        y = self.dense4(y)
        y = torch.relu(y)
        if training:
            y = self.dropout(y)
        y = y + residual
        y = self.dense5(y)
        return y


class Bigtwo312fNumpy:
    def __init__(self, state_dict):
        self.weights1 = np.array(state_dict["dense1.weight"]).T
        self.bias1 = np.array(state_dict["dense1.bias"]).T
        self.weights2 = np.array(state_dict["dense2.weight"]).T
        self.bias2 = np.array(state_dict["dense2.bias"]).T
        self.weights3 = np.array(state_dict["dense3.weight"]).T
        self.bias3 = np.array(state_dict["dense3.bias"]).T
        self.weights4 = np.array(state_dict["dense4.weight"]).T
        self.bias4 = np.array(state_dict["dense4.bias"]).T
        self.weights5 = np.array(state_dict["dense5.weight"]).T
        self.bias5 = np.array(state_dict["dense5.bias"]).T

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x):
        x = np.dot(x, self.weights1) + self.bias1
        x = self.relu(x)
        x = np.dot(x, self.weights2) + self.bias2
        x = self.relu(x)
        x = np.dot(x, self.weights3) + self.bias3
        x = self.relu(x)
        x = np.dot(x, self.weights4) + self.bias4
        x = self.relu(x)
        x = np.dot(x, self.weights5) + self.bias5

        return x

    def act(self, x):
        x = self.forward(x)
        return np.argmax(x)


class Agent312f:
    def __init__(self, device):
        self.histories = []
        self.rewards = []
        self.device = torch.device(device)
        self.model = Bigtwo312f(device)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=0.0001, alpha=0.99, momentum=0.0, eps=1e-5
        )

    def act(self, game, training=True):
        if len(self.histories) < len(self.rewards) + 1:
            self.histories.append([])
        obs = self.observe(game)
        actions = game.players[game.player_to_act].legal_actions
        if not training or torch.rand(1) > 0.01:
            with torch.no_grad():
                output = self.model(obs["x_batch"])
            action_index = torch.argmax(output, dim=0)[0]
        else:
            action_index = game.np_random.choice(len(actions))
        self.histories[-1].append(obs["x_batch"][action_index])
        action = actions[action_index]
        game.step(action)

    def learn(self):
        losses = []
        for i, history in enumerate(self.histories):
            self.optimizer.zero_grad()
            x_batch = torch.stack(history)
            output = self.model(x_batch)
            y_batch = torch.ones(x_batch.shape[0], 1).to(self.device) * self.rewards[i]
            loss = torch.nn.functional.mse_loss(output, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
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

        return dict(
            x_batch=x_batch,
        )

    def get_reward(self, game: Bigtwo, index):
        def punish(p):
            if p < 10:
                return p
            elif p < 13:
                return 2 * p
            else:
                return 3 * p

        if game.winner == index:
            lefts = [np.sum(p.holding) for p in game.players]
            lefts = [punish(p) for p in lefts]
            lefts = np.sum(lefts)
            reward = lefts * 0.1
            self.rewards.append(reward)
        else:
            lefts = np.sum(game.players[index].holding)
            lefts = punish(lefts)
            reward = -lefts * 0.1
            self.rewards.append(reward)

    def save(self, id="default"):
        checkpoint(self.model, self.optimizer, f"bigtwo312f-{id}")
