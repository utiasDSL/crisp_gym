"""Script showcasing how to record data in Lerobot Format."""

import argparse  # noqa: I001
import shutil
import time
from pathlib import Path
import PIL.Image  # noqa: F401

from crisp_py.gripper import Gripper, GripperConfig
from crisp_py.robot import Robot
import numpy as np

from crisp_py.camera import FrankaCameraConfig
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame
from rich import print

import crisp_gym  # noqa: F401
from crisp_gym.lerobot_wrapper import get_features
from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.manipulator_env_config import FrankaEnvConfig
from crisp_gym.record.recording_manager import RecordingManager


parser = argparse.ArgumentParser(description="Record data in Lerobot Format")
parser.add_argument(
    "--repo-id", type=str, default="franka_single", help="Repository ID for the dataset"
)
parser.add_argument("--task", type=str, default="pick the lego block.", help="Task description")
parser.add_argument("--robot-type", type=str, default="franka", help="Type of robot being used")
parser.add_argument("--fps", type=int, default=30, help="Frames per second for recording")
parser.add_argument(
    "--max-duration", type=float, default=30.0, help="Maximum episode duration in seconds"
)
parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to record")
args = parser.parse_args()

repo_id = args.repo_id
single_task = args.task
robot_type = args.robot_type
fps = args.fps
max_episode_duration = args.max_duration
num_episodes = args.num_episodes

# Clean up existing dataset if it exists
if Path(HF_LEROBOT_HOME / repo_id).exists():
    shutil.rmtree(HF_LEROBOT_HOME / repo_id)

# Wrist camera
camera_config_wrist = FrankaCameraConfig()
camera_config_wrist.camera_name = "right_wrist_camera"
camera_config_wrist.camera_color_image_topic = "right_wrist_camera/color/image_rect_raw"
camera_config_wrist.camera_color_info_topic = "right_wrist_camera/color/camera_info"

# Tp camera
camera_config_tp = FrankaCameraConfig()
camera_config_tp.camera_name = "right_third_person_camera"
camera_config_tp.camera_color_image_topic = "right_third_person_camera/color/image_raw"
camera_config_tp.camera_color_info_topic = "right_third_person_camera/color/camera_info"

env_config = FrankaEnvConfig(camera_configs=[camera_config_wrist, camera_config_tp])
env_config.gripper_enabled = True

path = Path("../crisp_py/config/gripper_right.yaml").resolve()
env_config.gripper_config = GripperConfig.from_yaml(path=path)
env_config.gripper_config.joint_state_topic = "gripper/gripper_state_broadcaster/joint_states"
env_config.gripper_config.command_topic = "gripper/gripper_position_controller/commands"
env_config.gripper_continous_control = True

env = ManipulatorCartesianEnv(namespace="right", config=env_config)
features = get_features(env)

dataset = LeRobotDataset.create(
    repo_id=repo_id,
    fps=fps,
    robot_type=robot_type,
    features=features,
    use_videos=False,
)

# Leader
leader = Robot(namespace="left")
leader.wait_until_ready()

path = Path("../crisp_py/config/trigger.yaml").resolve()
leader_gripper = Gripper(
    gripper_config=GripperConfig.from_yaml(path=path),
    namespace="left/gripper",
    index=1,
)
leader_gripper.wait_until_ready()
leader_gripper.value

# %% Prepare environment and leader

env.home()
env.reset()

env.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")

leader.wait_until_ready()
leader.cartesian_controller_parameters_client.load_param_config(
    file_path="../crisp_py/config/control/gravity_compensation.yaml"
)
leader.home()
leader.controller_switcher_client.switch_controller("cartesian_impedance_controller")

# %% Start interaction

start_time = -1
step = 0
# duration = 30  # seconds

previous_pose = leader.end_effector_pose
with RecordingManager(num_episodes=num_episodes) as recording_manager:
    while not recording_manager == "exit":
        print(
            f"[magenta bold]=== Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes} ==="
        )

        if recording_manager.state == "is_waiting":
            print("[magenta]Waiting for user to start.")
            while recording_manager.state == "is_waiting":
                time.sleep(1.0)

        print("[blue]Started episode")

        start_time = time.time()
        step = 0
        obs = {}
        while recording_manager.state == "recording":
            # Investigate timestamps:
            step_time_init = time.time()

            if step == 0:
                previous_pose = leader.end_effector_pose
                obs, _, _, _, _ = env.step(
                    action=np.concatenate(
                        [
                            np.zeros(6),
                            [
                                np.clip(
                                    leader_gripper.value
                                    if leader_gripper.value is not None
                                    else 0.0,
                                    0.0,
                                    1.0,
                                )
                            ],
                        ]
                    ),
                    block=False,
                )

                sleep_time = 1 / 30.0 - (time.time() - step_time_init)
                if sleep_time > 0:
                    time.sleep(sleep_time)  # Sleep to allow the environment to process the action
                    step += 1
                continue

            # print("Actual time is: ", time.time() - start_time)
            # print("Step time is: ", env.timestep * 1 / 30)

            action_pose = leader.end_effector_pose - previous_pose
            previous_pose = leader.end_effector_pose

            action = np.concatenate(
                [
                    action_pose.position,
                    action_pose.orientation.as_euler("xyz"),
                    np.array(
                        [
                            np.clip(
                                leader_gripper.value if leader_gripper.value is not None else 0.0,
                                a_min=0.0,
                                a_max=1.0,
                            )
                        ]
                    ),
                ]
            )

            action_dict = {dim: action[i] for i, dim in enumerate(features["action"]["names"])}
            obs_dict = {
                dim: obs["cartesian"][i] if i < 6 else obs["gripper"][0]
                for i, dim in enumerate(features["observation.state"]["names"])
            }
            # print(obs_dict)
            cam_frame = {
                f"observation.images.{camera.config.camera_name}": obs[
                    f"{camera.config.camera_name}_image"
                ]
                for camera in env.cameras
            }
            action_frame = build_dataset_frame(features, action_dict, prefix="action")
            obs_frame = build_dataset_frame(features, obs_dict, prefix="observation.state")

            frame = {**obs_frame, **action_frame, **cam_frame}
            dataset.add_frame(frame, task=single_task)

            obs, _, _, _, _ = env.step(action, block=False)

            sleep_time = 1 / 30.0 - (time.time() - step_time_init)
            if sleep_time > 0:
                time.sleep(sleep_time)  # Sleep to allow the environment to process the action
            step += 1

        if recording_manager.state == "paused":
            print(
                "[blue] Stopped episode. Waiting for user to decide whether to save or delete the episode"
            )
            # Reset funcionality to reset the robot and environment
            leader.home()
            leader.controller_switcher_client.switch_controller("cartesian_impedance_controller")
            env.home()
            env.reset()
            start_time = -1
            previous_pose = leader.end_effector_pose
            step = 0
        while recording_manager.state == "paused":
            time.sleep(1.0)

        if recording_manager.state == "to_be_saved":
            print("[green]Saving episode.")
            dataset.save_episode()
            recording_manager.episode_count += 1
            recording_manager.set_to_wait()

        if recording_manager.state == "to_be_deleted":
            print("[red]Deleting episode")
            dataset.clear_episode_buffer()
            recording_manager.set_to_wait()

        if recording_manager.state == "exit":
            break

dataset.push_to_hub(repo_id=repo_id)

leader.home()
leader.shutdown()

env.home()
env.close()
