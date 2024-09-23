import random

from env.env import Observation


class RandomAgent:

    def __init__(self):
        self.name = "Random"

    def act(self, obs: Observation):
        return random.choice(obs.legal_actions)
