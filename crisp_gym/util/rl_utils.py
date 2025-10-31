import time
import numpy as np
import json


def load_actions_safe(name: str) -> list[np.ndarray]:
    with open(f"/home/linusschwarz/crisp_gym/trajectories/{name}", "r") as f:
        actions = np.array(json.load(f))

    print(f"Loaded trajectory from {name}")
    return actions

def custom_reset(env, pos: list | np.ndarray, seed=None, options=None) -> tuple[dict, dict]:
    env.home()
    print("Homed.")
    obs, _ = env.reset()
    time.sleep(0.5)
    obs, *_ = env.step(np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0]), block=True)
    while np.linalg.norm(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3]) > 0.003:
        obs, *_ = env.step(np.zeros(7), block=True)
    time.sleep(0.5)
    print("Reached starting state.")
    obs, info = env.reset(seed=seed, options=options)
    time.sleep(0.5)
    return obs, info



def custom_reset_with_playback(env, pos: list | np.ndarray, action_sequence: list[np.ndarray], seed=None, options=None) -> tuple[dict, dict]:
    env.home()
    obs, _ = env.reset()
    time.sleep(0.5)
    print("Homed.")
    
    obs, *_ = env.step(np.array([pos[0], pos[1], pos[2], 0.0, 0.0, 0.0, 0.0]), block=True)
    while np.linalg.norm(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3]) > 0.003:
        obs, *_ = env.step(np.zeros(7), block=True)
    time.sleep(0.5)
    print("Reached starting state.")

    for act in action_sequence:
        obs, *_ = env.step(act, block=True)
    time.sleep(0.5)
    print("Executed reset-sequence")

    obs, info = env.reset(seed=seed, options=options)
    time.sleep(0.5)
    return obs, info
