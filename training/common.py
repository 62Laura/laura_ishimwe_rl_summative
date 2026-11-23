"""Helpers shared by every training script (DQN, REINFORCE, PPO, Actor-Critic)."""
import csv
import os
import random

import numpy as np
import torch

from environment.custom_env import MedicineDroneEnv


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def evaluate(act_fn, episodes=30, seed=10_000):
    """Play fresh episodes with the greedy policy. act_fn maps a numpy observation to an action.
    Returns the mean return, the crash rate and the mean number of boxes delivered."""
    env = MedicineDroneEnv()
    returns, crashes, boxes = [], 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(act_fn(obs))
            total += r
            done = term or trunc
        returns.append(total); crashes += info["crashed"]; boxes.append(info["boxes_delivered"])
    return dict(eval_return=float(np.mean(returns)), eval_crash_rate=crashes / episodes,
                eval_boxes=float(np.mean(boxes)))


def save_log(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
