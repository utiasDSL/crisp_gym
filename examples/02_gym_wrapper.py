"""Draw a circle using the ManipulatorCartesianEnv with the RecedingHorizon and WindowWrapper."""

import numpy as np

from crisp_gym.env_wrapper import RecedingHorizon, WindowWrapper
from crisp_gym.envs.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.envs.manipulator_env_config import FrankaEnvConfig

# === Circle Parameters ===
RADIUS = 0.1  # [m]
CENTER = np.array([0.4, 0.0, 0.4])
CTRL_FREQ = 50  # control frequency in Hz
SIN_FREQ = 0.25  # frequency of circular motion in Hz
ITERATIONS = 5  # number of full circles to draw

# === Environment Settings ===
ACTION_HORIZON = 5
WINDOW_SIZE = 2

# === Environment Setup ===
env_config = FrankaEnvConfig(control_frequency=CTRL_FREQ)
env = ManipulatorCartesianEnv(namespace="right", config=env_config)
env = RecedingHorizon(env, horizon_length=ACTION_HORIZON)
env = WindowWrapper(env, window_size=WINDOW_SIZE)

# === Move to Starting Point ===
start_position = CENTER + [0, RADIUS, 0]
print(f"Moving to start position: {start_position}")
env.move_to(position=start_position, speed=0.15)
env.gripper.open()
obs, _ = env.reset()

# === Generate Circle Trajectory ===
time_period = ITERATIONS / SIN_FREQ  # total time in seconds
steps = int(time_period * CTRL_FREQ)

angles = 2 * np.pi * SIN_FREQ * np.arange(steps) / CTRL_FREQ
x = RADIUS * np.sin(angles)
y = RADIUS * np.cos(angles)

# Velocity (finite difference)
dx = np.diff(np.concatenate([[x[-1]], x]))
dy = np.diff(np.concatenate([[y[-1]], y]))

print(f"Drawing circle for {ITERATIONS} iterations with {steps} steps.")

# === Execute Circular Motion ===
for t in range(0, steps, ACTION_HORIZON):
    action = np.zeros((ACTION_HORIZON, 7))

    idxs = (t + np.arange(ACTION_HORIZON)) % steps
    action[:, 0] = dx[idxs]
    action[:, 1] = dy[idxs]

    # Close gripper after half of the trajectory
    if t > steps / 2:
        action[:, 6] = 1.0

    obs, _, _, _, _ = env.step(action, block=True)

    print(f"Observations at step {t}:")
    for obs_name, obs_value in obs.items():
        print(f"{obs_name}: {obs_value.shape}")
    print("-" * 40)

print("Circle drawing complete.")

env.home()
env.close()
