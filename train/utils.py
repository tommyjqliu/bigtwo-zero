import typing
import logging
import traceback
import torch

from env.env import BigtwoEnv

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    )
)
log = logging.getLogger("doudzero")
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def get_batch(free_queue, full_queue, buffers, flags, lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch


def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    positions = range(4)
    optimizers = {}
    for position in positions:
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            momentum=flags.momentum,
            eps=flags.epsilon,
            alpha=flags.alpha,
        )
        optimizers[position] = optimizer
    return optimizers


def create_buffers(flags, device):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """
    T = flags.unroll_length
    positions = range(4)
    buffers = {}

    for position in positions:
        specs = dict(
            done=dict(size=(T,), dtype=torch.bool),
            episode_return=dict(size=(T,), dtype=torch.float32),
            target=dict(size=(T,), dtype=torch.float32),
            obs_x_no_action=dict(size=(T, 52), dtype=torch.int8),
            obs_action=dict(size=(T, 52), dtype=torch.int8),
            obs_z=dict(size=(T, 5, 156), dtype=torch.int8),
        )
        _buffers: Buffers = {key: [] for key in specs}
        for _ in range(flags.num_buffers):
            for key in _buffers:
                if not device == "cpu":
                    _buffer = (
                        torch.empty(**specs[key])
                        .to(torch.device(device))
                        .share_memory_()
                    )
                else:
                    _buffer = (
                        torch.empty(**specs[key])
                        .to(torch.device("cpu"))
                        .share_memory_()
                    )
                _buffers[key].append(_buffer)
        buffers[position] = _buffers
    return buffers


def act(i, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """
    positions = range(4)
    try:
        T = flags.unroll_length
        log.info("Actor %i started.", i)

        env = BigtwoEnv()

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        obs_x_no_action_buf = {p: [] for p in positions}
        obs_action_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}

        while True:
            obs = env.reset().to_tensor("cuda:0")
            while not obs.done:
                if len(obs.legal_actions) != 1:
                    p = obs.player_to_act
                    obs_x_no_action_buf[p].append(obs.x)
                    obs_z_buf[p].append(obs.z)
                    with torch.no_grad():
                        agent_output = model.forward(
                            p, obs.z_batch, obs.x_batch, flags=flags
                        )
                    action_idx = int(agent_output["action"].cpu().detach().numpy())
                    action = obs.legal_actions[action_idx]
                    obs_action_buf[p].append(torch.tensor(action.code, dtype=torch.float32))
                    size[p] += 1
                else: # passing
                    action = obs.legal_actions[0]

                obs = env.step(action).to_tensor("cuda:0")

            for p in positions:
                diff = size[p] - len(target_buf[p])
                if diff > 0:
                    done_buf[p].extend([False for _ in range(diff - 1)])
                    done_buf[p].append(True)
                    episode_return = 1 if p == obs.winner else -0.33
                    # episode_return = -1 if p == obs.winner else 1
                    # episode_return = 1 if env.game.players[p].holding.sum() != 13 else -1
                    # print("finished with", env.game.players[p].holding.sum())
                    # if p != obs.winner:
                    #     episode_return += (8 - env.game.players[p].holding.sum()) * 0.1
                    episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                    episode_return_buf[p].append(episode_return)
                    target_buf[p].extend([episode_return for _ in range(diff)])

            for p in positions:
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]["done"][index][t, ...] = done_buf[p][t]
                        buffers[p]["episode_return"][index][t, ...] = (
                            episode_return_buf[p][t]
                        )
                        buffers[p]["target"][index][t, ...] = target_buf[p][t]
                        buffers[p]["obs_x_no_action"][index][t, ...] = (
                            obs_x_no_action_buf[p][t]
                        )
                        buffers[p]["obs_action"][index][t, ...] = obs_action_buf[p][t]
                        buffers[p]["obs_z"][index][t, ...] = obs_z_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    obs_x_no_action_buf[p] = obs_x_no_action_buf[p][T:]
                    obs_action_buf[p] = obs_action_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error("Exception in worker process %i", i)
        traceback.print_exc()
        print()
        raise e
