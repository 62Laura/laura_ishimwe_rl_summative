# Medicine Drone: DQN, REINFORCE, PPO and A2C with TorchRL

**Mission:** solve everyday problems in Africa through technology. In this game a **medicine delivery drone** keeps four rural clinics stocked from a central hub, while managing its battery, hills (more battery) and drifting storms (much more battery).

## The environment (`environment/custom_env.py`)
| | |
|---|---|
| **Actions (5)** | UP, DOWN, LEFT, RIGHT, SERVICE (at the hub: reload and recharge; at a clinic: hand over a box) |
| **Observation (27 numbers)** | drone position, battery, load, direction to the hub, each clinic's offset and stock, storm positions, cost of the 4 neighbouring cells, time of day |
| **Rewards** | +10 per unit of stock delivered, small progress bonus for flying towards the goal, small step and empty-clinic costs, -0.2 for wasted actions, -5 if the drone runs out of battery in the air |
| **Start state** | random clinics, hills and storms; the drone starts at the hub, loaded, battery 50 |
| **End** | the day ends after 200 steps, or the drone is lost |

Rendering (`environment/rendering.py`) is a retro pygame view of the map.

## Install and run
```bash
uv sync
uv run main.py                              # best DQN agent: GUI window + terminal log
uv run main.py --algo ppo --episodes 5      # dqn, reinforce, ppo, a2c, heuristic or random
uv run main.py --algo dqn --no-gui          # terminal only
uv run pytest                               # environment tests
```

## Training (TorchRL)
Each algorithm has 10 runs with different hyperparameters (listed in `CONFIGS` in the training files).
```bash
uv run python -m training.dqn_training --run 1
uv run python -m training.pg_training --algo ppo --run 1      # reinforce, ppo or a2c
```
Models are saved in `models/dqn/` and `models/pg/` (the best of each as `best_*.pt`), logs and result tables in `logs/`, plots in `assets/`.

## Structure
```
main.py                 run an agent with GUI and terminal output
environment/            custom_env.py (the game), rendering.py (pygame)
training/               dqn_training.py, pg_training.py (REINFORCE, PPO, A2C), common.py
models/dqn, models/pg   saved networks
logs/                   training logs, hyperparameter tables
assets/                 plots and pictures
tests/                  environment tests
```
