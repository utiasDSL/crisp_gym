"""Script showcasing how to record data in Lerobot Format."""

import argparse  # noqa: I001
import os
import pathlib
import shutil
import time
from pathlib import Path

import numpy as np
import PIL.Image  # noqa: F401
from crisp_py.camera import FrankaCameraConfig
from crisp_py.gripper import GripperConfig
from crisp_py.gripper.gripper import Gripper
from crisp_py.robot import Robot
from crisp_py.robot_config import FrankaConfig
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame
from rich import print

import crisp_gym  # noqa: F401
from crisp_gym.lerobot_wrapper import get_features
from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.manipulator_env_config import FrankaEnvConfig
from crisp_gym.record.recording_manager import RecordingManager

# Set ROS environment variables
# Get the script's directory and workspace root
script_dir = pathlib.Path(__file__).parent.resolve()
workspace_root = script_dir.parent.parent
os.environ['ROS_DOMAIN_ID'] = '100'
os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp'
os.environ['CYCLONEDDS_URI'] = f"file://{script_dir / 'cyclone_config.xml'}"

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

# Gripper 
gripper_config_follower = GripperConfig(
    min_value=0.0,
    max_value=1.0,
    command_topic="gripper/gripper_position_controller/commands",
    joint_state_topic="gripper/joint_states"
)

# Gripper config for leader
gripper_config_leader = GripperConfig(
    min_value = 0.0, 
    max_value = 1.0,
    command_topic = "gripper/gripper_position_controller/commands",
    joint_state_topic = "gripper/joint_states"
)

# Follower
env_config = FrankaEnvConfig(camera_configs=[camera_config_wrist, camera_config_tp], gripper_config=gripper_config_follower)
env = ManipulatorCartesianEnv(namespace="right", config=env_config)
features = get_features(env)

# Leader
faster_publishing_config = FrankaConfig()  
faster_publishing_config.publish_frequency = 200.0
leader = Robot(robot_config=faster_publishing_config, namespace="left")

# Initialize leader gripper & wait for it to be ready
leader_gripper = Gripper(namespace="left", gripper_config=gripper_config_leader)
leader_gripper.wait_until_ready()

# Wait for follower gripper to be ready
env.gripper.wait_until_ready()

dataset = LeRobotDataset.create(
    repo_id=repo_id,
    fps=fps,
    robot_type=robot_type,
    features=features,
    use_videos=False,
)

env.home()
env.reset()
env.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")

leader.wait_until_ready()
leader.cartesian_controller_parameters_client.load_param_config(
    file_path=str(workspace_root / 'crisp_py' / 'config' / 'control' / 'gravity_compensation.yaml')
)
leader.home()
leader.controller_switcher_client.switch_controller("cartesian_impedance_controller")

start_time = -1


# %%
def sync(leader, env, leader_gripper):  # noqa: ANN001, D103
    env.robot.set_target(pose=leader.end_effector_pose)
    if leader_gripper.value is not None:
        env.gripper.set_target(leader_gripper.value)
    

leader.node.create_timer(
    1.0 / faster_publishing_config.publish_frequency, lambda: sync(leader, env, leader_gripper)
)

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

        while recording_manager.state == "recording":
            # NOTE: This might look wrong but while doing kinesthetic teaching, the action is the reached
            # state after steping. This is the position where the operator moved the robot arm to.
            obs_pre_step = env._get_obs()
            obs_after_step, _, _, _, _ = env.step(np.array([0.0] * 7))

            # Investigate timestamps:
            if start_time == -1:
                start_time = time.time()
            # print("Actual time is: ", time.time() - start_time)
            # print("Step time is: ", env.step * fps)

            action = np.concatenate(
                (
                    obs_after_step["cartesian"] - obs_pre_step["cartesian"],
                    np.array([obs_after_step["gripper"][0]]),
                ),
                axis=0,
            )

            action_dict = {dim: action[i] for i, dim in enumerate(features["action"]["names"])}
            obs_dict = {
                dim: obs_pre_step["cartesian"][i] if i < 6 else float(obs_pre_step["gripper"][0])
                for i, dim in enumerate(features["observation.state"]["names"])
            }

            cam_frame = {
                f"observation.images.{camera.config.camera_name}": obs_pre_step[
                    f"{camera.config.camera_name}_image"
                ]
                for camera in env.cameras
            }
            action_frame = build_dataset_frame(features, action_dict, prefix="action")
            obs_frame = build_dataset_frame(features, obs_dict, prefix="observation.state")

            frame = {**obs_frame, **action_frame, **cam_frame}
            dataset.add_frame(frame, task=single_task)
            # time.sleep(1 / fps)

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

leader.home()
env.home()
leader.shutdown()
env.close()
