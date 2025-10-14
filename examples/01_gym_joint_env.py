"""Control the joints using the ManipulatorJointEnv interactively."""

import numpy as np

from crisp_gym.envs.manipulator_env import ManipulatorJointEnv
from crisp_gym.envs.manipulator_env_config import FrankaEnvConfig

# === Configuration ===
START_POSITION = np.array([0.4, 0.0, 0.4])
ROT_STEP = 0.1
N_JOINTS = 7
CTRL_FREQ = 10

# === Initialize environment ===
env_config = FrankaEnvConfig(control_frequency=CTRL_FREQ)
env = ManipulatorJointEnv(namespace="right", config=env_config)

print("Moving to start position...")
env.move_to(position=START_POSITION, speed=0.15)
env.gripper.open()
obs, _ = env.reset()

# === Control Loop ===
current_joint = 0
print("\nControl the robot joints with your keyboard:")
print("Controls: [a] rotate -, [d] rotate +, [s] switch axis, [e] exit\n")

while True:
    try:
        key = input(f"[Joint {current_joint}] Command (a/d/s/e): ").strip().lower()
        action = np.zeros((8))

        if key == "a":
            action[current_joint] = -ROT_STEP
        elif key == "d":
            action[current_joint] = ROT_STEP
        elif key == "s":
            current_joint = (current_joint + 1) % N_JOINTS
            print(f"Switched to joint {current_joint}.")
            continue
        elif key == "e":
            break
        else:
            print("Invalid key. Use 'a', 'd', 's', or 'e'.")
            continue

        obs, _, _, _, _ = env.step(action, block=False)
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting joint control.")
        break


print("Returning to home position...")
env.home()
env.close()
print("Environment closed.")
