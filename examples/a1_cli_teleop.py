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
env = make_env("my_env_v2", namespace="left")

print("Env created")
env.wait_until_ready()
print("Env ready.")

if "--no-reset" not in sys.argv [1:]:
    obs, _ = custom_reset(env, [0.200, -0.020, -0.200])
else:
    env.gripper.close()
    obs, _ = env.reset()

all_actions = []

if len(sys.argv) > 1 and sys.argv[1] != "--no-reset":
    actions = load_actions_safe(sys.argv[1])
    for act in actions:
        obs ,*_ = env.step(act, block=True)
        all_actions.append(np.copy(act))
        if len(sys.argv) > 2 and sys.argv[2] == "--print":
            with np.printoptions(precision=2, suppress=True):
                print(f"Pos: {(obs['observation.state.cartesian'][:3])*1000} mm; Target: {(obs['observation.state.target'][:3])*1000} mm; Error: {(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3])*1000} mm;",
                        f"Gripper: Pos: {obs['observation.state.gripper']*100} %; Target: {obs['observation.target.gripper']*100} %; Error: {(obs['observation.state.gripper'] - obs['observation.target.gripper'])*100} %")
        # velocity = np.linalg.norm(act[:3])
        # n_steps_pos = 1 #  np.ceil(velocity / max_step_velocity)
        # n_steps_grip = np.ceil(np.abs(act[-1]) / max_gripper_step)
        # n_steps = np.maximum(np.maximum(n_steps_pos, n_steps_grip), 1.0)

        # for _ in range(int(n_steps)):
        #     obs ,*_ = env.step(act / n_steps, block=True)
        #     all_actions.append(act / n_steps)
        # if np.linalg.norm(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3]) > max_step_velocity:
        #     last_obs = obs
        #     while np.linalg.norm(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3]) > max_step_velocity:
        #         obs, *_ = env.step(np.zeros(7), block=True)
        #         all_actions.append(np.zeros(7))
        #         # with np.printoptions(precision=2, suppress=True):
        #         #     print(f"did {(np.linalg.norm(obs['observation.state.cartesian'][:3] - last_obs['observation.state.cartesian'][:3])) * 1000} mm progress")
        #         last_obs = obs


print("Starting teleop")
# last_obs = obs["observation.state.cartesian"][:3]
try: 
    while True:
        action = np.zeros(7)
        inp = input("Next command: ")
        obs, _, _, _, _ = env.step(action, block=True)
        if len(inp) == 0:
            continue
        if inp == "q":
            break
        if inp.startswith("p"):
            with np.printoptions(precision=2, suppress=True):
                print(f"Pos: {(obs['observation.state.cartesian'][:3])*1000} mm; Target: {(obs['observation.state.target'][:3])*1000} mm; Error: {(obs['observation.state.cartesian'][:3] - obs['observation.state.target'][:3])*1000} mm;",
                      f"Gripper: Pos: {obs['observation.state.gripper']*100} %; Target: {obs['observation.target.gripper']*100} %; Error: {(obs['observation.state.gripper'] - obs['observation.target.gripper'])*100} %")
            if len(inp) > 1 and inp[1] == "s":
                last_obs = obs["observation.state.cartesian"][:3]
            continue
        if inp == "obs":
            print(f"obs: {obs}")
        if inp == "ros":
            print("Topics:", env.robot.node.get_topic_names_and_types())
        if inp == "z":
            with np.printoptions(precision=2, suppress=True):
                print(f"Delta: {(obs['observation.state.cartesian'][:3] - last_obs)*1000} mm")
            if len(inp) > 1 and inp[1] == "s":
                last_obs = obs["observation.state.cartesian"][:3]
            continue
        if inp[0] == "#":
            for coordinate in inp[1:].split(" "):
                axis, value = coordinate[0], coordinate[1:]
                if axis == "x":
                    action[0] = float(value) * 0.001
                elif axis == "y":
                    action[1] = float(value) * 0.001
                elif axis == "z":
                    action[2] = float(value) * 0.001
                else:
                    print(f"Unknown axis: {axis}")
                    continue
        if inp[0] == "h":
            env.home()
            obs, _ = env.reset()
            time.sleep(1)
            all_actions.clear()
            continue
        if inp == "img":
            cv2.imwrite("temp.png", cv2.cvtColor(obs["observation.images.wrist_camera"], cv2.COLOR_RGB2BGR))
        if inp == "dep":
                depth_m = obs["observation.images.wrist_depth_camera"]  # in m
                max_depth_m = 0.3  # 30 cm
                depth_8u = np.clip(depth_m, 0, max_depth_m)  # Clip to max depth
                depth_8u = (depth_8u / max_depth_m * 255).astype(np.uint8)

                # 2. Colorize
                colored = cv2.applyColorMap(depth_8u, cv2.COLORMAP_PLASMA)

                # 3. Save image
                cv2.imwrite("temp_depth.png", colored)
        else:
            for c in inp:
                if c == "f":
                    action[0] += 0.001
                elif c == "F":
                    action[0] += 0.025
                elif c == "b":
                    action[0] -= 0.001
                elif c == "B":
                    action[0] -= 0.025
                elif c == "l":
                    action[1] += 0.001
                elif c == "L":
                    action[1] += 0.025
                elif c == "r":
                    action[1] -= 0.001
                elif c == "R":
                    action[1] -= 0.025
                elif c == "u":
                    action[2] += 0.001
                elif c == "U":
                    action[2] += 0.025
                elif c == "d":
                    action[2] -= 0.001
                elif c == "D":
                    action[2] -= 0.025
                elif c == "c":
                    action[6] = -0.2
                elif c == "o":
                    action[6] = 0.2

        magnitude = np.linalg.norm(action[:3])
        obs, _, _, _, _ = env.step(action, block=True)
        all_actions.append(np.copy(action))

except KeyboardInterrupt as i:
    env.close()
    sys.exit(0)


file_name = f"trajectories/acts_{time.strftime('%Y%m%d-%H%M%S')}.json"
with open(file_name, "w") as f:
    json.dump(list(map(list, all_actions)), f)
print(f"Saved {len(all_actions)} actions to {file_name}")

print("Going back home.")
env.home()
print("Homed.")

if input("Playback? (y/n)") == "y":
    obs, _ = custom_reset(env, [0.200, -0.020, -0.200])
    for act in all_actions:
        velocity = np.linalg.norm(act[:3])
        n_steps_pos = np.ceil(velocity / max_step_velocity)
        n_steps_grip = np.ceil(np.abs(act[-1]) / max_gripper_step)
        n_steps = np.maximum(np.maximum(n_steps_pos, n_steps_grip), 1.0)
        for _ in range(int(n_steps)):
            obs ,*_ = env.step(act / n_steps, block=True)

print("Going back home.")
env.home()
print("Homed.")
env.close()

# 3.0752587e-01,  1.4205652e-04,  4.8634282e-01
# 3.11061263e-01,  1.09783876e-04,  4.39315856e-01,
# 3.0972812e-01,  6.8447778e-05,  4.3932381e-01,