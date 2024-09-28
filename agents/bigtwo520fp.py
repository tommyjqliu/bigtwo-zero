import numpy as np
import torch
from torch import nn
from env.game import Bigtwo
from utils.checkpoint import checkpoint
import numpy as np


class Bigtwo520fp(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)
        self.init_encoder()
        self.lstm = nn.LSTM(16 * 4, 32, batch_first=True)
        # shape: (batch, seq_len, input_size)
        self.dense1 = nn.Linear(32 + 16 * 6, 256)
        # lstm + holding + others_holding + other played + legal_actions
        self.dense2 = nn.Linear(256, 224)
        self.dense3 = nn.Linear(224, 128)
        self.dense4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)

        self.predicts = [
            nn.Sequential(
                nn.Linear(224, 128),
                nn.ReLU(),
                nn.Linear(128, 16),
            )
            for _ in range(3)
        ]

        self.to(self.device)

    def init_encoder(self):
        self.encode_dense1 = nn.Linear(52, 52)
        self.encode_dense2 = nn.Linear(52, 16)

    def encoder(self, x):
        x = self.encode_dense1(x)
        x = torch.relu(x)
        x = self.encode_dense2(x)
        x = torch.relu(x)
        return x

    def forward(self, x, z, training=True):
        x_shape = x.shape
        x_flattened = x.view(-1, 52)
        x_encoded = self.encoder(x_flattened)
        x_encoded = x_encoded.view(*x_shape[:-1], 16 * 6)

        z_shape = z.shape
        z_flattened = z.view(-1, 52)
        z_encoded = self.encoder(z_flattened)
        z_encoded = z_encoded.view(*z_shape[:-1], 16 * 4)

        lstm_out, (h_n, _) = self.lstm(z_encoded)
        lstm_out = lstm_out[:, -1, :]

        holdings = None
        y = torch.cat([lstm_out, x_encoded], dim=-1)
        y = self.dense1(y)
        y = torch.relu(y)
        if training:
            y = self.dropout(y)
        y = self.dense2(y)
        y = torch.relu(y)
        if training:
            y = self.dropout(y)
            holdings = [predict(y) for predict in self.predicts]
            holdings = torch.stack(holdings, dim=1)
        y = self.dense3(y)
        y = torch.relu(y)
        if training:
            y = self.dropout(y)
        y = self.dense4(y)

        return y, holdings


class Bigtwo520fNumpy:
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


class Agent520fp:
    def __init__(self, device):
        self.x_histories = []
        self.z_histories = []
        self.holdings_histories = []
        self.rewards = []
        self.device = torch.device(device)
        self.model = Bigtwo520fp(device)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=0.0001, alpha=0.99, momentum=0.0, eps=1e-5
        )

    def act(self, game, training=True):
        if len(self.x_histories) < len(self.rewards) + 1:
            self.x_histories.append([])
            self.z_histories.append([])
            self.holdings_histories.append([])

        obs = self.observe(game)
        actions = game.players[game.player_to_act].legal_actions
        if not training or torch.rand(1) > 0.01:
            with torch.no_grad():
                y, preds = self.model(obs["x_batch"], obs["z_batch"], training=False)
            action_index = torch.argmax(y, dim=0)[0]
        else:
            action_index = game.np_random.choice(len(actions))

        self.x_histories[-1].append(obs["x_batch"][action_index])
        self.z_histories[-1].append(obs["z_batch"][action_index])

        if training:
            with torch.no_grad():
                encode_holdings = self.model.encoder(obs["others_holding"])
            self.holdings_histories[-1].append(encode_holdings)

        action = actions[action_index]
        game.step(action)

    def learn(self):
        losses = []
        for i, x_history in enumerate(self.x_histories):
            self.optimizer.zero_grad()
            x_batch = torch.stack(x_history)
            z_batch = torch.stack(self.z_histories[i])
            holdings_batch = torch.stack(self.holdings_histories[i])
            known_rounds = z_batch.sum(dim=1).bool().sum(dim=1)

            pred_y_batch, pred_holdings = self.model(x_batch, z_batch)
            y_batch = torch.ones(x_batch.shape[0], 1).to(self.device) * self.rewards[i]
            reward_loss = torch.nn.functional.mse_loss(pred_y_batch, y_batch)
            holdings_loss = torch.nn.functional.mse_loss(pred_holdings, holdings_batch)
            loss = reward_loss + holdings_loss * known_rounds * 0.1

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
            self.optimizer.step()
            losses.append(loss.detach())
        self.x_histories = []
        self.z_histories = []
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
        all_others_holding = np.bitwise_not(np.bitwise_or(player.holding, all_played))
        all_others_holding = torch.tensor(all_others_holding).to(self.device)
        others_played = [history[::-1][i::4] for i in range(4)][:3]
        others_played = [
            np.bitwise_or.reduce(h, axis=0) if h else empty for h in others_played
        ]
        others_played = np.concatenate(others_played)
        others_played = torch.tensor(others_played).to(self.device)
        others_holding = [players[(player_to_act + i) % 4].holding for i in range(1, 4)]
        others_holding = torch.tensor(others_holding).to(self.device)

        x = torch.cat([holding, all_others_holding, others_played], dim=0)
        x_batch = x.repeat(len(legal_actions), 1)
        x_batch = torch.cat([x_batch, legal_actions], dim=1).float()

        last_16_actions = np.zeros((16, 52), dtype=bool)
        for i, action in enumerate(history[::-1][:16]):
            last_16_actions[i] = action
        z = last_16_actions[::-1].reshape(-1, 208)
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
            others_holding=others_holding,
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
        checkpoint(self.model, self.optimizer, f"bigtwo520-{id}")
