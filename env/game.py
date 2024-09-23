import numpy as np
from .utils import (
    Action,
    Player,
)


class Bigtwo:
    def __init__(self):
        self.np_random = np.random.RandomState()
        self.reset()

    def reset(self):
        self.traces = []
        initial_decks = np.arange(52)
        self.np_random.shuffle(initial_decks)
        self.player_to_act = 0
        self.player_in_charge = None
        self.action_in_charge = None
        self.players: list[Player] = []
        self.winner = None

        for i in range(4):
            handcard = np.zeros(52, dtype=bool)
            indices = initial_decks[i * 13 : (i + 1) * 13]
            handcard[indices] = True
            player = Player(handcard)

            if player.holding[0]:
                self.player_to_act = i
                player.legal_actions = []
                for action in player.all_actions:
                    if action.code[0]:
                        player.legal_actions.append(action)

            self.players.append(player)

    def step(self, action: Action):
        player = self.players[self.player_to_act]
        player.history.append(action.code)
        self.traces.append((self.player_to_act, action.code))
        if not action.is_pass:
            player.update(action.code)
            self.action_in_charge = action
            self.player_in_charge = self.player_to_act
            if np.sum(player.holding) == 0:
                self.winner = self.player_to_act
                return

        self.player_to_act = (self.player_to_act + 1) % 4
        next_player = self.players[self.player_to_act]
        if self.player_to_act == self.player_in_charge:
            self.player_in_charge = None
            self.action_in_charge = None
            next_player.legal_actions = next_player.all_actions
        else:
            next_player.pre_action(self.action_in_charge)
