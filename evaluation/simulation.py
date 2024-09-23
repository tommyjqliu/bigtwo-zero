from env.env import BigtwoEnv
from .deep_agent import DeepAgent
from .random_agent import RandomAgent


def evaluate():
    players = [
        DeepAgent("bigtwo_checkpoints/bigtwo/1_weights_153600.ckpt"),
        RandomAgent(),
        RandomAgent(),
        RandomAgent(),
    ]

    game_num = 100
    wins = [0, 0, 0, 0]
    env = BigtwoEnv()
    for i in range(game_num):
        while env.game.winner is None:
            player = players[env.game.player_to_act]
            action = player.act(env.observe())
            env.step(action)
        wins[env.game.winner] += 1
        env.reset()
    
    print(wins)
