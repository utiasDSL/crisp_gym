import copy
import sys
import time
import numpy as np
from crisp_gym.manipulator_env import ManipulatorCartesianEnv, make_env_config
from crisp_gym.util.rl_utils import load_actions_safe, custom_reset

max_step_velocity = 0.01
max_gripper_step = 0.05

config = make_env_config("my_env")
print("Env created.")
env = ManipulatorCartesianEnv(config=config)
env.wait_until_ready()
print("Env ready.")

actions = load_actions_safe(sys.argv[1])
do_print = len(sys.argv) > 2 and sys.argv[2] == "--print"

obs, _ = custom_reset(env, [0.200, -0.020, -0.200])


class PlaybackActionSource:
    def __init__(self, trajectory: list[np.ndarray]):
        self.trajectory = trajectory
        self.i = 0
        # self.sub = []
        self.current_action = np.zeros(7)
        pass

    def reset(self):
        self.i = 0
        self.current_action = np.zeros(7)
        # self.sub = []

    def next(self, obs) -> np.ndarray | None:
        clip_act = max_step_velocity
        if np.linalg.norm(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3]) > max_step_velocity:
            return np.zeros(7)
        if np.any(self.current_action != 0.0):
            next_action = np.clip(self.current_action, np.array([-clip_act, -clip_act, -clip_act, 0.0, 0.0, 0.0, -max_gripper_step]), np.array([clip_act, clip_act, clip_act, 0.0, 0.0, 0.0, max_gripper_step]))
            self.current_action -= next_action
            return next_action

        # if len(self.sub) == 0:
        #     if self.i >= len(self.trajectory):
        #         return None
        #     act = self.trajectory[self.i]
        #     self.i += 1
            # velocity = np.linalg.norm(act[:3])
            # n_steps_pos = np.ceil(velocity / 0.03)
            # n_steps_grip = np.ceil(np.abs(act[-1]) / max_gripper_step)
            # n_steps = np.maximum(np.maximum(n_steps_pos, n_steps_grip), 1.0)
            # self.sub = [act / n_steps for _ in range(int(n_steps))]
    #    return self.sub.pop()

        if self.i >= len(self.trajectory):
            return None
        act = self.trajectory[self.i]
        self.i += 1
        self.current_action = act
        next_action = np.clip(self.current_action, np.array([-clip_act, -clip_act, -clip_act, 0.0, 0.0, 0.0, -max_gripper_step]), np.array([clip_act, clip_act, clip_act, 0.0, 0.0, 0.0, max_gripper_step]))
        self.current_action -= next_action
        return next_action

 

# action_source = PlaybackActionSource(actions)
# while (act := action_source.next(obs)) is not None:
#     if do_print:
#          print(list(act))
#     obs, *_ = env.step(act, block=True)

for act in actions:
    obs, *_ = env.step(act, block=True)


# for act in actions:
#     velocity = np.linalg.norm(act[:3])  
#     n_steps_pos = np.ceil(velocity / 0.02)
#     n_steps_grip = np.ceil(np.abs(act[-1]) / max_gripper_step)
#     n_steps = np.maximum(np.maximum(n_steps_pos, n_steps_grip), 1.0)
#     for _ in range(int(n_steps)):
#         obs ,*_ = env.step(act / n_steps, block=True)
#         if do_print:
#             print(act / n_steps)
#     if np.linalg.norm(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3]) > max_step_velocity:
#         last_obs = obs
#         while np.linalg.norm(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3]) > max_step_velocity:
#             obs, *_ = env.step(np.zeros(7), block=True)
#             if do_print:
#                 print(np.zeros(7))
#             # with np.printoptions(precision=2, suppress=True):
#             #     print(f"did {(np.linalg.norm(obs['observation.state.cartesian'][:3] - last_obs['observation.state.cartesian'][:3])) * 1000} mm progress")
#             last_obs = obs

print("Going back home.")
env.home()
print("Homed.")
env.close()
