"""Draw a circle using the ManipulatorCartesianEnv."""

import time
import numpy as np
import matplotlib.pyplot as plt

from crisp_gym.manipulator_env import ManipulatorCartesianEnv, make_env, make_env_config
from crisp_gym.manipulator_env_config import FrankaEnvConfig
import gc

# %% === Circle Parameters ===
RADIUS = 0.1  # [m]
CENTER = np.array([0.5, -0.15, 0.3])
CTRL_FREQ = 15  # control frequency in Hz
SIN_FREQ = 0.1  # frequency of circular motion in Hz


# %% === Environment Setup ===
env = make_env("my_env")
env.wait_until_ready()

# %% === Move to Starting Point ===
start_position = CENTER + [0, RADIUS, 0]
print(f"Moving to start position: {start_position}")
env.move_to(position=start_position, speed=0.15)
time.sleep(1.0)
env.gripper.open()
obs, _ = env.reset()

# Prepare plotting
plt.ion()
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Target vs Real EE XY Positions')

# %%=== Generate Circle Trajectory ===
steps = int(CTRL_FREQ / SIN_FREQ)

angles = np.linspace(0, 2 * np.pi, steps, endpoint=False)
x = RADIUS * np.sin(angles)
y = RADIUS * np.cos(angles)

# Velocity (finite difference)
dx = np.diff(np.concatenate([[x[-1]], x]))
dy = np.diff(np.concatenate([[y[-1]], y]))


# %% === Execute Circular Motion ===
try:
    gc.disable()
    while True:
        t0 = time.perf_counter()
        ts = [t0]
        obs_dts = []
        cam_dts = []
        sleep_dts = []
        step_dts = []
        outside_dts = []

        # initialize target/real position traces for this circle
        target_positions = []
        real_positions = []
        actual_target_positions = []
        t_after = None

        # start from current known start position
        current_target = np.array(start_position, dtype=float)
        target_positions.append(current_target.copy())

        # initial real position from observation (cartesian first 3 entries)

        real_positions.append(np.array(obs["observation.state.cartesian"][:3], dtype=float))
        actual_target_positions.append(np.array(obs["observation.state.target"][:3], dtype=float))

        for t in range(steps):
            action = np.zeros((7))

            action[0] = dx[t]
            action[1] = dy[t]

            # update target by adding the delta (assumes dx/dy are in same frame)
            current_target = current_target + np.array([dx[t], dy[t], 0.0])
            target_positions.append(current_target.copy())

            # Close gripper after half of the trajectory
            if t > steps / 2:
                action[6] = 1.0

            pre_step_time = time.perf_counter()
            if t_after is not None:
                outside_dts.append(pre_step_time - t_after)
            obs, _, _, _, info = env.step(action, block=True)
            t_after = time.perf_counter()
            step_dts.append(t_after - pre_step_time)
            ts.append(t_after)
            obs_dts.append(info["obs_time"])
            cam_dts.append(obs["dt_camera"])
            sleep_dts.append(info["sleep_time"])

            real_positions.append(np.array(obs["observation.state.cartesian"][:3], dtype=float))
            actual_target_positions.append(np.array(obs["observation.state.target"][:3], dtype=float))

            # print(obs)
        delta = time.perf_counter() - t0
        dts = np.diff(ts)
        dt_std = np.std(dts)
        obs_dt_std = np.std(obs_dts)
        cam_dt_std = np.std(cam_dts)
        sleep_dt_std = np.std(sleep_dts)
        step_dt_std = np.std(step_dts)
        outside_dt_std = np.std(outside_dts)
        print(f"Completed circle of {steps=} in {delta:.2f} seconds, effective freq: {steps / delta:.2f} Hz, std dev: {dt_std*1000:.2f} ms, std dev (obs): {obs_dt_std*1000:.2f} ms, std dev (cam): {cam_dt_std*1000:.2f} ms, std dev (slp): {sleep_dt_std*1000:.2f} ms, std dev (stp): {step_dt_std*1000:.2f} ms, std dev (out): {outside_dt_std*1000:.2f} ms")

        # Plot XY trajectories
        targ = np.array(target_positions)
        real = np.array(real_positions)
        act_targ = np.array(actual_target_positions)

        ax.clear()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('Target vs Real EE XY Positions')
        ax.plot(targ[:, 0], targ[:, 1], '-r', label='ideal target')
        ax.plot(real[:, 0], real[:, 1], '-b', label='real')
        ax.plot(act_targ[:, 0], act_targ[:, 1], '-g', label='actual target')
        ax.scatter([targ[0, 0]], [targ[0, 1]], c='green', marker='x', label='start')
        ax.legend()
        plt.draw()
        plt.pause(0.01)
        obs, _ = env.reset()
        gc.collect()
except KeyboardInterrupt:
    print("\nCircle drawing interrupted by user.")
    print("Going back home.")
    env.home()

    env.close()





# print("Move to center position.")
# env.move_to(position=CENTER, speed=0.15)
# ROT_STEP = 0.05
# AXIS_MAP = ['x', 'y', 'z']
# current_axis = 0
#
#
# print("\n--- Manual EEF Rotation ---")
# print("Controls: [a] rotate -, [d] rotate +, [s] switch axis, [e] exit\n")
#
# while True:
#    try:
#        key = input(f"Current axis: {AXIS_MAP[current_axis]} | Input (a/d/s/e): ").strip().lower()
#        action = np.zeros((7))
#
#        if key == "a":
#            action[3 + current_axis] = -ROT_STEP
#        elif key == "d":
#            action[3 + current_axis] = ROT_STEP
#        elif key == "s":
#            current_axis = (current_axis + 1) % 3
#            print(f"Switched to axis: {AXIS_MAP[current_axis]}")
#            continue
#        elif key == "e":
#            print("Exiting rotation control.")
#            break
#        else:
#            print("Invalid input. Use a/d/s/e.")
#            continue
#
#        obs, _, _, _, _ = env.step(action, block=False)
#
#    except KeyboardInterrupt:
#        print("\nInterrupted. Exiting rotation control.")
#        break


