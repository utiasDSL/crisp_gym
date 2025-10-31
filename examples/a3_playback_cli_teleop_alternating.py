import sys
import time
import numpy as np
from crisp_gym.manipulator_env import ManipulatorCartesianEnv, make_env_config
from crisp_gym.util.rl_utils import custom_reset, load_actions_safe

max_step_velocity = 0.005
max_gripper_step = 0.05

config = make_env_config("my_env")
env = ManipulatorCartesianEnv(config=config)
print("Env created")
env.wait_until_ready()
print("Env ready.")

actions_1 = load_actions_safe(sys.argv[1])
actions_2 = load_actions_safe(sys.argv[2])


i = 1
try:
    while True:
        obs, _ = custom_reset(env, [0.200, -0.020, -0.200])
        
        print(f"Playing trajectory 1 [{i}]")
        for act in actions_1:
            velocity = np.linalg.norm(act[:3])
            n_steps_pos = np.ceil(velocity / max_step_velocity)
            n_steps_grip = np.ceil(np.abs(act[-1]) / max_gripper_step)
            n_steps = np.maximum(np.maximum(n_steps_pos, n_steps_grip), 1.0)
            for _ in range(int(n_steps)):
                obs ,*_ = env.step(act / n_steps, block=True)


        obs, _ = custom_reset(env, [0.200, -0.020, -0.200])

        print(f"Playing trajectory 2 [{i}]")
        for act in actions_2:
            velocity = np.linalg.norm(act[:3])
            n_steps_pos = np.ceil(velocity / max_step_velocity)
            n_steps_grip = np.ceil(np.abs(act[-1]) / max_gripper_step)
            n_steps = np.maximum(np.maximum(n_steps_pos, n_steps_grip), 1.0)
            for _ in range(int(n_steps)):
                obs ,*_ = env.step(act / n_steps, block=True)

        i += 1
except KeyboardInterrupt:
    print("Interrupted by user.")
    env.close()
