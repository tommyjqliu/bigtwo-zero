import numpy as np
from .game import Bigtwo
import torch


class BigtwoEnv:
    def __init__(self):
        self.game = Bigtwo()

    def reset(self):
        self.game.reset()
        return Observation(self)

    def observe(self):
        return Observation(self)

    def step(self, action):
        self.game.step(action)
        return Observation(self)


class Observation:
    def __init__(self, env: BigtwoEnv):
        players = env.game.players
        player_to_act = env.game.player_to_act
        self.legal_actions = players[player_to_act].legal_actions
        self.legal_action_codes = np.array(
            [a.code for a in self.legal_actions], dtype=bool
        )

        holding = players[player_to_act].holding
        indices = [
            (i + player_to_act) % 4
            for i in range(4)
            if (i + player_to_act) % 4 != player_to_act
        ]
        others_holding = [players[i].holding for i in indices]
        others_holding = np.bitwise_or.reduce(others_holding, axis=0)
        action_in_charge = env.game.action_in_charge
        others_played = [players[i].played for i in indices]
        others_played = np.concatenate(others_played, axis=0)
        others_left = [np.sum(players[i].holding) for i in indices]
        others_left = [np.eye(13, dtype=bool)[i - 1] for i in others_left]
        others_left = np.concatenate(others_left, axis=0)
        last_15_actions = [np.zeros(52, dtype=bool) for _ in range(15)]
        self.x = np.concatenate(
            [
                holding,
                others_holding,
                # action_in_charge.code if action_in_charge else np.zeros(52, dtype=bool),
                # others_played,
                # others_left,
            ]
        )

        l = len(env.game.traces[-15:])
        for i in range(l):
            player, action = env.game.traces[-i - 1]
            last_15_actions[i] = action
        self.z = np.concatenate(last_15_actions).reshape(5, -1)

        self.x_batch = np.repeat(
            self.x[np.newaxis, :], len(self.legal_action_codes), axis=0
        )
        self.x_batch = np.concatenate(
            [self.x_batch, self.legal_action_codes],
            axis=1,
        )

        self.z_batch = np.repeat(
            self.z[np.newaxis, :], len(self.legal_action_codes), axis=0
        )

        self.player_to_act = env.game.player_to_act
        self.winner = env.game.winner
        self.done = env.game.winner is not None

    def to_tensor(self, device):
        return TensorObservation(self, device)


class TensorObservation:
    def __init__(self, obs: Observation, device):
        self.player_to_act = obs.player_to_act
        self.winner = obs.winner
        self.done = obs.done
        self.legal_actions = obs.legal_actions

        self.x = torch.tensor(obs.x, dtype=torch.float32).to(device)
        self.z = torch.tensor(obs.z, dtype=torch.float32).to(device)
        self.x_batch = torch.tensor(obs.x_batch, dtype=torch.float32).to(device)
        self.z_batch = torch.tensor(obs.z_batch, dtype=torch.float32).to(device)
        self.legal_action_codes = torch.tensor(
            obs.legal_action_codes, dtype=torch.float32
        ).to(device)
