import numpy as np
from gymnasium.utils.env_checker import check_env
from environment.custom_env import MedicineDroneEnv, UP, DOWN, LEFT, RIGHT, LOAD, DROP, RECHARGE


def run(policy, episodes=100, seed=0):
    env = MedicineDroneEnv()
    totals, crashes, stockouts = [], 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            obs, r, term, trunc, info = env.step(policy(obs, env))
            total += r
            done = term or trunc
        totals.append(total); crashes += info["crashed"]; stockouts.append(info["stockout_steps"])
    return float(np.mean(totals)), crashes, float(np.mean(stockouts))


def random_policy(obs, env):
    return env.action_space.sample()


def _towards(env, tx, ty):
    dx, dy = tx - env.dx, ty - env.dy
    if abs(dx) >= abs(dy) and dx != 0:
        return RIGHT if dx > 0 else LEFT
    return DOWN if dy > 0 else UP


def heuristic_policy(obs, env):
    """Reload at the hub, then fly to the clinic with the lowest stock and drop a box."""
    at_hub = (env.dx, env.dy) == env.hub
    if at_hub:
        if env.battery < env.max_battery - 1:
            return RECHARGE
        if env.load < env.capacity:
            return LOAD
    if env.load == 0:
        return _towards(env, *env.hub)
    k = int(np.argmin(env.stock))
    if (env.dx, env.dy) == (env.cx[k], env.cy[k]):
        return DROP
    dist_back = abs(env.cx[k] - env.hub[0]) + abs(env.cy[k] - env.hub[1])
    if env.battery < 2.2 * dist_back + 4:         # not enough battery to get there and back
        return _towards(env, *env.hub)
    return _towards(env, env.cx[k], env.cy[k])


def test_gym_api():
    check_env(MedicineDroneEnv(), skip_render_check=True)


def test_heuristic_beats_random():
    r, h = run(random_policy), run(heuristic_policy)
    print(f"random {r} | heuristic {h}   (mean return, crashes, mean stockout steps)")
    assert h[0] > r[0] + 5
