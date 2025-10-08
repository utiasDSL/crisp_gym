"""Draw a circle using the ManipulatorCartesianEnv."""

import numpy as np

from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.manipulator_env_config import FrankaEnvConfig
from crisp_py.gripper.gripper import GripperConfig
from crisp_py.camera.camera import CameraConfig

# %% === Circle Parameters ===
RADIUS = 0.1  # [m]
CENTER = np.array([0.4, 0.0, 0.4])
CTRL_FREQ = 50  # control frequency in Hz
SIN_FREQ = 0.25  # frequency of circular motion in Hz
ITERATIONS = 5  # number of full circles to draw

gripper_config = GripperConfig(
    min_value=0.0,
    max_value=1.0,
    command_topic="/gripper/gripper_position_controller/commands",
    joint_state_topic="/gripper/joint_states",
    max_delta=10.0,
)
camera_configs = [
    CameraConfig(
        camera_name="primary",
        camera_frame="primary_link",
        resolution=(256, 256),
        camera_color_image_topic="/camera/camera/color/image_rect_raw",
        camera_color_info_topic="/camera/camera/color/camera_info",
    )
]

# %% === Environment Setup ===
env_config = FrankaEnvConfig(
    control_frequency=CTRL_FREQ,
    gripper_config=gripper_config,
    camera_configs=camera_configs,
)
env = ManipulatorCartesianEnv(config=env_config)

# %% === Move to Starting Point ===
start_position = CENTER + [0, RADIUS, 0]
print(f"Moving to start position: {start_position}")
env.move_to(position=start_position, speed=0.15)
env.gripper.open()
obs, _ = env.reset()

# %%=== Generate Circle Trajectory ===
time_period = ITERATIONS / SIN_FREQ  # total time in seconds
steps = int(time_period * CTRL_FREQ)

angles = 2 * np.pi * SIN_FREQ * np.arange(steps) / CTRL_FREQ
x = RADIUS * np.sin(angles)
y = RADIUS * np.cos(angles)

# Velocity (finite difference)
dx = np.diff(np.concatenate([[x[-1]], x]))
dy = np.diff(np.concatenate([[y[-1]], y]))

print(f"Drawing circle for {ITERATIONS} iterations with {steps} steps.")

# %% === Execute Circular Motion ===
for t in range(steps):
    action = np.zeros((7))

    idxs = t % steps
    action[0] = dx[idxs]
    action[1] = dy[idxs]

    # Close gripper after half of the trajectory
    if t > steps / 2:
        action[6] = 1.0

    obs, _, _, _, _ = env.step(action, block=True)

print("Circle drawing complete.")

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


print("Going back home.")
env.home()

env.close()
