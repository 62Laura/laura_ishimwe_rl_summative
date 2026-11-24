"""Run a trained agent in the Medicine Drone game, with the GUI window and a live terminal log.

Examples:
    uv run main.py                          # best DQN agent, 3 episodes, GUI + terminal
    uv run main.py --algo ppo --episodes 5
    uv run main.py --algo heuristic         # rule-based reference agent
    uv run main.py --algo random --no-gui   # no window, terminal only
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
from environment.custom_env import MedicineDroneEnv  # noqa: E402

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "SERVICE"]
MODELS = {"dqn": "models/dqn/best_dqn.pt", "reinforce": "models/pg/best_reinforce.pt",
          "ppo": "models/pg/best_ppo.pt", "a2c": "models/pg/best_a2c.pt"}


def load_agent(algo):
    """Return a function observation -> action (the greedy action of the saved network)."""
    if algo == "random":
        rng = np.random.default_rng(0)
        return lambda obs, env: int(rng.integers(5))
    if algo == "heuristic":
        from tests.test_env import heuristic_policy
        return lambda obs, env: heuristic_policy(obs, env)
    from torchrl.modules import MLP
    path = MODELS[algo]
    if not os.path.exists(path):
        sys.exit(f"Model file not found: {path}. Train it first (see README).")
    ckpt = torch.load(path, map_location="cpu")
    net = MLP(in_features=ckpt["obs_dim"], out_features=ckpt["n_actions"], num_cells=[ckpt["hidden"]] * 2)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    def act(obs, env):
        with torch.no_grad():
            return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())
    return act


def main():
    p = argparse.ArgumentParser(description="Run a trained agent in the Medicine Drone game.")
    p.add_argument("--algo", default="dqn", choices=["dqn", "reinforce", "ppo", "a2c", "heuristic", "random"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=10_000, help="first episode seed (10000 = the evaluation seeds)")
    p.add_argument("--fps", type=int, default=8, help="speed of the window (frames per second)")
    p.add_argument("--no-gui", action="store_true", help="terminal only, no window")
    p.add_argument("--verbose", action="store_true", help="print every step, not only the key events")
    args = p.parse_args()

    act = load_agent(args.algo)
    env = MedicineDroneEnv(render_mode=None if args.no_gui else "human")
    env.metadata["render_fps"] = args.fps
    print(f"\n=== Medicine Drone | agent: {args.algo.upper()} | episodes: {args.episodes} ===")
    print("Goal: keep the 4 clinics stocked with medicine, never run out of battery.\n")

    scores, boxes, crashes = [], [], 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        env.render()
        done, total = False, 0.0
        print(f"--- Episode {ep + 1} (seed {args.seed + ep}) ---")
        while not done:
            a = act(obs, env)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
            env.render()
            if args.verbose or env.last_event:
                print(f"step {env.t:3d} | {ACTIONS[a]:7s} | reward {r:+6.2f} | battery {env.battery:4.0f} "
                      f"| load {env.load} | boxes {info['boxes_delivered']:2d}"
                      + (f" | {env.last_event}" if env.last_event else ""))
        scores.append(total); boxes.append(info["boxes_delivered"]); crashes += info["crashed"]
        print(f"Episode {ep + 1} finished: return {total:.1f} | boxes delivered {info['boxes_delivered']} "
              f"| empty-clinic steps {info['stockout_steps']} | {'DRONE LOST' if info['crashed'] else 'drone safe'}\n")
        time.sleep(0.5)
    env.close()
    print(f"=== Summary ({args.algo.upper()}, {args.episodes} episodes) ===")
    print(f"mean return {np.mean(scores):.1f} | mean boxes {np.mean(boxes):.1f} | crashes {crashes}/{args.episodes}")


if __name__ == "__main__":
    main()
