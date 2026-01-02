import copy
import json
import sys
import time
import cv2
import numpy as np
from crisp_gym.manipulator_env import ManipulatorCartesianEnv, make_env, make_env_config
from crisp_gym.util.rl_utils import custom_reset, load_actions_safe

# config = make_env_config("my_env_v2")

max_step_velocity = 0.005
max_gripper_step = 0.05

# env = ManipulatorCartesianEnv(config=config, namespace="left")
env = make_env("my_env_v3_grav_comp")

print("Env created")
env.wait_until_ready()
print("Env ready.")


obs, _ = env.reset()

input()


print("Going back home.")
env.home()
print("Homed.")
env.close()

# 3.0752587e-01,  1.4205652e-04,  4.8634282e-01
# 3.11061263e-01,  1.09783876e-04,  4.39315856e-01,
# 3.0972812e-01,  6.8447778e-05,  4.3932381e-01,
