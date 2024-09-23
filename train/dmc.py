import os
import threading
import time
import timeit
import pprint
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .models import Model
from .utils import get_batch, log, create_buffers, create_optimizers, act

mean_episode_return_buf = {p: deque(maxlen=100) for p in range(4)}


def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss


def learn(position, actor_model, learn_model, batch, optimizer, flags, lock):
    """Performs a learning (optimization) step."""
    if flags.training_device != "cpu":
        device = torch.device("cuda:" + str(flags.training_device))
    else:
        device = torch.device("cpu")
    obs_x_no_action = batch["obs_x_no_action"].to(device)
    obs_action = batch["obs_action"].to(device)
    obs_x = torch.cat((obs_x_no_action, obs_action), dim=2).float()
    obs_x = torch.flatten(obs_x, 0, 1)
    obs_z = torch.flatten(batch["obs_z"].to(device), 0, 1).float()
    target = torch.flatten(batch["target"].to(device), 0, 1)
    episode_returns = batch["episode_return"][batch["done"]]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        optimizer.zero_grad()
        # target[obs_x[:, 351:403].sum(dim=1) == 0] = -1
        # print("obs_x", obs_x[0, 351:403], obs_x.shape, "target", target[0])
        learner_outputs = learn_model(obs_x, return_value=True)
        loss = compute_loss(learner_outputs["values"], target)
        stats = {
            "mean_episode_return_"
            + str(position): torch.mean(
                torch.stack([_r for _r in mean_episode_return_buf[position]])
            ).item(),
            "loss_" + str(position): loss.item(),
        }
        loss.backward()
        nn.utils.clip_grad_norm_(learn_model.parameters(), flags.max_grad_norm)
        optimizer.step()

        actor_model.get_model(position).load_state_dict(learn_model.state_dict())
        return stats


def train(flags):
    T = flags.unroll_length
    B = flags.batch_size
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    actor_model = Model(device=flags.training_device)
    actor_model.share_memory()
    actor_model.eval()
    buffers = create_buffers(flags, "cuda:0")

    learner_model = Model(device=flags.training_device)
    optimizers = create_optimizers(flags, learner_model)

    # pretrained = torch.load(
    #     "/home/tommy/repository/DouZero/bigtwo_checkpoints_0.56/bigtwo/3_weights_371200.ckpt",
    #     map_location="cuda:0",
    # )
    # for k in range(4):
    #     learner_model.get_model(k).load_state_dict(pretrained)
    # Initialize buffers

    # Initialize queues
    actor_processes = []
    ctx = mp.get_context("spawn")
    free_queue = {p: ctx.SimpleQueue() for p in range(4)}
    full_queue = {p: ctx.SimpleQueue() for p in range(4)}

    # Stat Keys
    stat_keys = [
        "mean_episode_return_0",
        "loss_0",
        "mean_episode_return_1",
        "loss_1",
        "mean_episode_return_2",
        "loss_2",
        "mean_episode_return_3",
        "loss_3",
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {p: 0 for p in range(4)}

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                i,
                free_queue,
                full_queue,
                actor_model,
                buffers,
                flags,
            ),
        )
        actor.start()
        actor_processes.append(actor)

    def batch_and_learn(i, position, local_lock, position_lock, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, position_frames, stats
        while frames < flags.total_frames:
            batch = get_batch(
                free_queue[position],
                full_queue[position],
                buffers[position],
                flags,
                local_lock,
            )
            _stats = learn(
                position,
                actor_model,
                learner_model.get_model(position),
                batch,
                optimizers[position],
                flags,
                position_lock,
            )

            with lock:
                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
                position_frames[position] += T * B

    for m in range(flags.num_buffers):
        for p in range(4):
            free_queue[p].put(m)

    threads = []
    locks = {p: threading.Lock() for p in range(4)}
    position_locks = {p: threading.Lock() for p in range(4)}

    for i in range(flags.num_threads):
        for position in range(4):
            thread = threading.Thread(
                target=batch_and_learn,
                name="batch-and-learn-%d" % i,
                args=(
                    i,
                    position,
                    locks[position],
                    position_locks[position],
                ),
            )
            thread.start()
            threads.append(thread)

    def checkpoint(frames):
        if flags.disable_checkpoint:
            return
        log.info("Saving checkpoint to %s", checkpointpath)
        _models = learner_model.get_models()
        torch.save(
            {
                "model_state_dict": {k: _models[k].state_dict() for k in _models},
                "optimizer_state_dict": {
                    k: optimizers[k].state_dict() for k in optimizers
                },
                "stats": stats,
                "flags": vars(flags),
                "frames": frames,
                "position_frames": position_frames,
            },
            checkpointpath,
        )

        # Save the weights for evaluation purpose
        for position in range(4):
            model_weights_dir = os.path.expandvars(
                os.path.expanduser(
                    "%s/%s/%s"
                    % (
                        flags.savedir,
                        flags.xpid,
                        str(position) + "_weights_" + str(frames) + ".ckpt",
                    )
                )
            )
            torch.save(
                learner_model.get_model(position).state_dict(), model_weights_dir
            )

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer() - flags.save_interval * 60
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()

            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {
                k: (position_frames[k] - position_start_frames[k])
                / (end_time - start_time)
                for k in position_frames
            }
            log.info(
                "After %i (0:%i 1:%i 2:%i 3:%i) frames: @ %.1f fps (avg@ %.1f fps) (0:%i 1:%i 2:%i 3:%i) Stats:\n%s",
                frames,
                *[position_frames[k] for k in position_frames],
                fps,
                fps_avg,
                *[position_fps[k] for k in position_fps],
                pprint.pformat(stats),
            )

    except KeyboardInterrupt:
        checkpoint(frames)
        return
    else:
        for thread in threads:
            thread.join()
        log.info("Learning finished after %d frames.", frames)

    checkpoint(frames)
    plogger.close()
