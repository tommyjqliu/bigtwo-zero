from env.game import Bigtwo


class RandomAgent:
    def __init__(self):
        self.histories = []
        self.rewards = []

    def act(self, game: Bigtwo):
        index = game.np_random.choice(
            len(game.players[game.player_to_act].legal_actions)
        )
        action = game.players[game.player_to_act].legal_actions[index]
        game.step(action)

    def learn(self):
        self.rewards = []
        self.histories = []
        return 0

    def save(self, id):
        pass

    def get_reward(self, game, index):
        pass
