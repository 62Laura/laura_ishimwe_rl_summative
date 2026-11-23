"""Policy-gradient methods on MedicineDroneEnv, built with TorchRL: REINFORCE, PPO and Actor-Critic (A2C).

TorchRL pieces used: GymWrapper (env), RewardSum/StepCounter (episode statistics), MLP + TensorDictModule +
ProbabilisticActor (the policy: logits -> categorical distribution), ValueOperator (the critic), Collector,
GAE (advantage estimation), ClipPPOLoss (PPO), A2CLoss (Actor-Critic) and ReplayBuffer (PPO mini-batches).
REINFORCE has no critic: it learns from the discounted return of whole episodes.

    uv run python -m training.pg_training --algo reinforce --run 1
"""
import os
import time
import warnings

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tensordict.nn import TensorDictModule
from torch.distributions import OneHotCategorical
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import Compose, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss
from torchrl.objectives.value import GAE

try:
    from torchrl.collectors import Collector                  # torchrl >= 0.15
except ImportError:
    from torchrl.collectors import SyncDataCollector as Collector

from environment.custom_env import MedicineDroneEnv
from training.common import evaluate, save_log, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------- hyperparameters
DEFAULTS = {
    "reinforce": dict(lr=1e-3, gamma=0.99, episodes_per_update=8, entropy_coef=0.01, normalize=True, hidden=128),
    "ppo": dict(lr=1e-3, gamma=0.99, lmbda=0.95, clip_eps=0.2, epochs=4, sub_batch=64, frames_per_batch=1024,
                entropy_coef=0.01, hidden=128),
    "a2c": dict(lr=7e-4, gamma=0.99, lmbda=0.95, frames_per_batch=256, entropy_coef=0.01, critic_coef=0.5, hidden=128),
}
COMMON = dict(total_frames=150_000, seed=0)

# The 10 runs of each algorithm (run number -> changes from the defaults above).
CONFIGS = {
    "reinforce": {
        1: dict(),                                    # baseline
        2: dict(lr=3e-4),                             # smaller learning rate
        3: dict(lr=3e-3),                             # larger learning rate
        4: dict(gamma=0.95),                          # shorter-sighted
        5: dict(gamma=0.90),                          # even shorter-sighted
        6: dict(episodes_per_update=4),               # noisier gradient (fewer episodes per update)
        7: dict(episodes_per_update=16),              # smoother gradient (more episodes per update)
        8: dict(entropy_coef=0.05),                   # more exploration
        9: dict(normalize=False),                     # no return normalisation (no baseline)
        10: dict(lr=3e-3, gamma=0.95, episodes_per_update=16),   # a combination
    },
    "ppo": {
        1: dict(),
        2: dict(lr=3e-4),
        3: dict(lr=1e-4),
        4: dict(gamma=0.95),
        5: dict(clip_eps=0.1),
        6: dict(clip_eps=0.3),
        7: dict(epochs=8),
        8: dict(frames_per_batch=2048),
        9: dict(entropy_coef=0.05),
        10: dict(lr=2e-3, gamma=0.95, epochs=8),
    },
    "a2c": {
        1: dict(),
        2: dict(lr=3e-4),
        3: dict(lr=2e-3),
        4: dict(gamma=0.95),
        5: dict(gamma=0.90),
        6: dict(frames_per_batch=128),
        7: dict(frames_per_batch=512),
        8: dict(entropy_coef=0.05),
        9: dict(lmbda=0.8),
        10: dict(lr=2e-3, gamma=0.95, frames_per_batch=512),
    },
}


def make_env():
    return TransformedEnv(GymWrapper(MedicineDroneEnv()), Compose(RewardSum(), StepCounter()))


def build_networks(env, hidden, critic=True):
    obs_dim = env.observation_spec["observation"].shape[-1]
    n_actions = env.action_spec.space.n
    logit_net = MLP(in_features=obs_dim, out_features=n_actions, num_cells=[hidden] * 2)
    actor = ProbabilisticActor(
        TensorDictModule(logit_net, in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"], spec=env.action_spec, distribution_class=OneHotCategorical, return_log_prob=True)
    value = None
    if critic:
        value = ValueOperator(MLP(in_features=obs_dim, out_features=1, num_cells=[hidden] * 2), in_keys=["observation"])
    return logit_net, actor, value, obs_dim, n_actions


def discounted_returns(rewards, gamma):
    out, running = torch.zeros_like(rewards), 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


# ---------------------------------------------------------------- REINFORCE
def run_reinforce(env, cfg):
    logit_net, actor, _, obs_dim, n_actions = build_networks(env, cfg["hidden"], critic=False)
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg["lr"])
    episodes, frames = [], 0
    while frames < cfg["total_frames"]:
        obs_l, act_l, ret_l = [], [], []
        for _ in range(cfg["episodes_per_update"]):
            with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                td = env.rollout(max_steps=1000, policy=actor, break_when_any_done=True)
            rewards = td["next", "reward"].squeeze(-1)
            obs_l.append(td["observation"]); act_l.append(td["action"])
            ret_l.append(discounted_returns(rewards, cfg["gamma"]))
            frames += td.shape[0]
            episodes.append((len(episodes) + 1, frames, td.shape[0], float(rewards.sum())))
        obs, act, G = torch.cat(obs_l), torch.cat(act_l), torch.cat(ret_l)
        if cfg["normalize"]:
            G = (G - G.mean()) / (G.std() + 1e-8)                  # baseline: subtract the mean return
        dist = actor.get_dist(env.reset().clone().update({"observation": obs}))
        logp = dist.log_prob(act)
        loss = -(logp * G).mean() - cfg["entropy_coef"] * dist.entropy().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return logit_net, episodes, obs_dim, n_actions


# ---------------------------------------------------------------- PPO and A2C
def run_actor_critic(env, cfg, algo):
    logit_net, actor, value, obs_dim, n_actions = build_networks(env, cfg["hidden"])
    collector = Collector(env, actor, frames_per_batch=cfg["frames_per_batch"], total_frames=cfg["total_frames"], device="cpu")
    advantage = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=value, average_gae=True)
    if algo == "ppo":
        loss_fn = ClipPPOLoss(actor, value, clip_epsilon=cfg["clip_eps"], entropy_bonus=True,
                              entropy_coeff=cfg["entropy_coef"], critic_coeff=1.0, loss_critic_type="smooth_l1")
        buffer = ReplayBuffer(storage=LazyTensorStorage(cfg["frames_per_batch"]), sampler=SamplerWithoutReplacement())
    else:
        loss_fn = A2CLoss(actor, value, entropy_bonus=True, entropy_coeff=cfg["entropy_coef"],
                          critic_coeff=cfg["critic_coef"], loss_critic_type="smooth_l1")
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])

    episodes, frames = [], 0
    for batch in collector:
        frames += batch.numel()
        done = batch["next", "done"].squeeze(-1)
        for ret, length in zip(batch["next", "episode_reward"][done].flatten().tolist(),
                               batch["next", "step_count"][done].flatten().tolist()):
            episodes.append((len(episodes) + 1, frames, length, ret))
        if algo == "ppo":
            for _ in range(cfg["epochs"]):
                advantage(batch)
                buffer.extend(batch.reshape(-1))
                for _ in range(cfg["frames_per_batch"] // cfg["sub_batch"]):
                    sub = buffer.sample(cfg["sub_batch"])
                    out = loss_fn(sub)
                    loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 1.0); optimizer.step()
        else:
            advantage(batch)
            out = loss_fn(batch.reshape(-1))
            loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 1.0); optimizer.step()
    collector.shutdown()
    return logit_net, episodes, obs_dim, n_actions


# ---------------------------------------------------------------- one run
def train_pg(algo, run_id, cfg, models_dir="models/pg", logs_dir="logs/pg"):
    cfg = {**DEFAULTS[algo], **COMMON, **cfg}
    set_seed(cfg["seed"])
    env = make_env()
    start = time.time()
    if algo == "reinforce":
        logit_net, episodes, obs_dim, n_actions = run_reinforce(env, cfg)
    else:
        logit_net, episodes, obs_dim, n_actions = run_actor_critic(env, cfg, algo)

    def act(obs):
        with torch.no_grad():
            return int(logit_net(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())

    result = evaluate(act)
    os.makedirs(models_dir, exist_ok=True)
    torch.save({"state_dict": logit_net.state_dict(), "obs_dim": obs_dim, "n_actions": n_actions,
                "hidden": cfg["hidden"], "algo": algo}, os.path.join(models_dir, f"{algo}_run{run_id:02d}.pt"))
    save_log(os.path.join(logs_dir, f"{algo}_run{run_id:02d}.csv"), episodes, ["episode", "frames", "length", "return"])
    last = [e[3] for e in episodes[-20:]]
    result.update(train_last20=float(np.mean(last)) if last else float("nan"), episodes=len(episodes),
                  minutes=round((time.time() - start) / 60, 1))
    return cfg, result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Train one policy-gradient run (run number 1..10 from CONFIGS).")
    parser.add_argument("--algo", choices=["reinforce", "ppo", "a2c"], required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--frames", type=int, default=COMMON["total_frames"])
    args = parser.parse_args()
    cfg, result = train_pg(args.algo, args.run, {**CONFIGS[args.algo][args.run], "total_frames": args.frames})
    row = {"algo": args.algo, "run": args.run, **cfg, **result}
    os.makedirs("logs/pg", exist_ok=True)
    with open(f"logs/pg/{args.algo}_run{args.run:02d}_result.json", "w") as f:
        json.dump(row, f, indent=2)
    print(json.dumps({k: row[k] for k in ("algo", "run", "eval_return", "eval_crash_rate", "eval_boxes", "train_last20", "minutes")}))
