import numpy as np
import torch
from env.game import Bigtwo
from models import Bigtwo104, Bigtwo156
from torch import tensor
from utils.checkpoint import checkpoint


class Agent:
    def __init__(self, device):
        self.device = device
        self.histories = []
        self.rewards = []

    def observe(self, game: Bigtwo):
        players = game.players
        player_to_act = game.player_to_act
        player = players[player_to_act]
        legal_actions = player.legal_actions
        legal_actions = tensor(np.array([a.code for a in legal_actions])).to(
            self.device
        )
        holding = tensor(player.holding).to(self.device)
        other_indices = [
            (i + player_to_act) % 4
            for i in range(4)
            if (i + player_to_act) % 4 != player_to_act
        ]
        others_holding = [players[i].holding for i in other_indices]
        others_holding = np.bitwise_or.reduce(others_holding, axis=0)
        others_holding = tensor(others_holding).to(self.device)
        return dict(
            legal_actions=legal_actions,
            holding=holding,
            others_holding=others_holding,
        )

    def act(self, game: Bigtwo):
        if len(self.histories) < len(self.rewards) + 1:
            self.histories.append([])

        obs = self.observe(game)
        with torch.no_grad():
            output = self.model(obs["x_batch"].float())
        action_index = torch.argmax(output, dim=0)[0]
        action = game.players[game.player_to_act].legal_actions[action_index]
        self.histories[-1].append(obs["x_batch"][action_index])
        game.step(action)

    def learn(self):
        assert len(self.histories) == len(self.rewards)
        losses = []
        for i, history in enumerate(self.histories):
            self.optimizer.zero_grad()
            x_batch = torch.stack(history)
            output = self.model(x_batch.float())
            y = (
                torch.tensor(self.rewards[i], dtype=torch.float32)
                .to(self.device)
                .repeat(len(history), 1)
            )
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.detach())
        self.histories = []
        self.rewards = []
        return torch.mean(torch.tensor(losses)).cpu().numpy()

    def save(self):
        checkpoint(self.model, self.optimizer, self.id)


class RandomAgent(Agent):
    def __init__(self):
        super().__init__("cpu")
        self.id = "random"

    def act(self, game: Bigtwo):
        actions = game.players[game.player_to_act].legal_actions
        index = np.random.randint(len(actions))
        action = actions[index]
        game.step(action)

    def learn(self):
        self.rewards = []
        self.histories = []
        return 0


class Agent104(Agent):
    def __init__(self, device):
        super().__init__(device)
        self.id = "104"
        self.model = Bigtwo104(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def observe(self, game: Bigtwo):
        obs = super().observe(game)
        x = torch.cat([obs["holding"]], dim=0)
        x_batch = x.repeat(len(obs["legal_actions"]), 1)
        x_batch = torch.cat([x_batch, obs["legal_actions"]], dim=1)
        return dict(
            x_batch=x_batch,
        )


class Agent156(Agent):
    def __init__(self, device):
        super().__init__(device)
        self.id = "156"
        self.model = Bigtwo156(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def observe(self, game: Bigtwo):
        obs = super().observe(game)
        x = torch.cat([obs["holding"], obs["others_holding"]], dim=0)
        x_batch = x.repeat(len(obs["legal_actions"]), 1)
        x_batch = torch.cat([x_batch, obs["legal_actions"]], dim=1)
        return dict(
            x_batch=x_batch,
        )
