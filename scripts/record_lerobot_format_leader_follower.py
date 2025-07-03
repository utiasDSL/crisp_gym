"""Script showcasing how to record data in Lerobot Format."""

import argparse  # noqa: I001
import os
import shutil
import time
from pathlib import Path
import PIL.Image  # noqa: F401

from crisp_py.gripper import Gripper, GripperConfig
from crisp_py.robot import Robot
import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame
from rich import print

import crisp_gym  # noqa: F401
from crisp_gym.lerobot_wrapper import get_features
from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.manipulator_env_config import AlohaFrankaEnvConfig
from crisp_gym.record.recording_manager import RecordingManager


parser = argparse.ArgumentParser(description="Record data in Lerobot Format")
parser.add_argument(
    "--repo-id",
    type=str,
    default="franka_single",
    help="Repository ID for the dataset",
)
parser.add_argument(
    "--task",
    type=str,
    default="pick the lego block.",
    help="Task description",
)
parser.add_argument(
    "--robot-type",
    type=str,
    default="franka",
    help="Type of robot being used",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Frames per second for recording",
)
parser.add_argument(
    "--max-duration",
    type=float,
    default=30.0,
    help="Maximum episode duration in seconds",
)
parser.add_argument(
    "--num-episodes",
    type=int,
    default=1,
    help="Number of episodes to record",
)
parser.add_argument(
    "--resume",
    type=bool,
    default=False,
    help="Resume recording of an already existing dataset",
)
parser.add_argument(
    "--push-to-hub",
    type=bool,
    default=True,
    help="Whether to push the dataset to the Hugging Face Hub.",
)
parser.add_argument(
    "--leader-controller",
    type=str,
    default="gravity_compensation_on_plane",
    help="Controller configuration for the leader robot.",
)
parser.add_argument(
    "--right-gripper-config",
    type=str,
    default="gripper_right",
    help="Gripper configuration for the right robot.",
)
parser.add_argument(
    "--trigger-config",
    type=str,
    default="trigger",
    help="Trigger configuration for the leader robot.",
)
# TODO: @maxdoesch add this
parser.add_argument(
    "--joint-control",
    type=bool,
    default=False,
    help="Whether to use joint control for the robot.",
)

args = parser.parse_args()


# Clean up existing dataset if it exists
if Path(HF_LEROBOT_HOME / args.repo_id).exists():
    shutil.rmtree(HF_LEROBOT_HOME / args.repo_id)

# Set up the config for the environment
path_to_config = os.environ.get("CRISP_CONFIG_PATH")
if path_to_config is None:
    raise ValueError(
        "You need to set the environment variable CRISP_CONFIG_PATH in order to load configs for the gripper and controller.\nTo do this execute export CRISP_CONFIG_PATH=path\\to\\config."
    )

# Set up the envionment configuration
gripper_config = GripperConfig.from_yaml(
    path=(Path(path_to_config) / (args.right_gripper_config + ".yaml")).resolve()
)
gripper_config.joint_state_topic = "gripper" + "/" + gripper_config.joint_state_topic
gripper_config.command_topic = "gripper" + "/" + gripper_config.command_topic

env_config = AlohaFrankaEnvConfig(gripper_config=gripper_config)
env = ManipulatorCartesianEnv(namespace="right", config=env_config)

# %% Prepare the Leader
leader = Robot(namespace="left")
leader.wait_until_ready()

leader_gripper_path = (Path(path_to_config) / (args.trigger_config + ".yaml")).resolve()
leader_gripper = Gripper(
    gripper_config=GripperConfig.from_yaml(path=leader_gripper_path),
    namespace="left/gripper",
)
leader_gripper.wait_until_ready()
leader_gripper.disable_torque()

# %% Prepare the dataset
features = get_features(env)

if args.resume:
    print(f"[green]Resuming recording from existing dataset: {args.repo_id}")
    dataset = LeRobotDataset(repo_id=args.repo_id)
else:
    print(f"[green]Creating new dataset: {args.repo_id}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=False,
    )

# %% Prepare environment and leader
env.home()
env.reset()

leader.home()
leader.cartesian_controller_parameters_client.load_param_config(
    file_path=Path(path_to_config) / "control" / (args.leader_controller + ".yaml")
)
leader.controller_switcher_client.switch_controller("cartesian_impedance_controller")

# %% Start interaction
previous_pose = leader.end_effector_pose
with RecordingManager(num_episodes=args.num_episodes) as recording_manager:
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
        is_first_step = True
        obs = {}

        # HACK: be sure that this one is running. Maybe the controller failed during recording so we need
        # to set the parameters again.
        # Reset before starting
        leader.reset_targets()
        leader.cartesian_controller_parameters_client.load_param_config(
            file_path=Path(path_to_config) / "control" / (args.leader_controller + ".yaml")
        )
        leader.controller_switcher_client.switch_controller("cartesian_impedance_controller")

        env.reset()

        while recording_manager.state == "recording":
            step_time_init = time.time()

            if is_first_step:
                previous_pose = leader.end_effector_pose
                # TODO: @danielsanjosepro make this steps clearer to the user
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

                is_first_step = False
                continue

            # print("Actual time is: ", time.time() - start_time)
            # print("Step time is: ", env.timestep * 1 / 30)

            # TODO: @danielsanjosepro make this steps clearer to the user
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

            # TODO: @danielsanjosepro or @maxdoesch make this saving in dict format more elegant
            action_dict = {dim: action[i] for i, dim in enumerate(features["action"]["names"])}
            obs_dict = {
                dim: obs["cartesian"][i] if i < 6 else obs["gripper"][0]
                for i, dim in enumerate(features["observation.state"]["names"])
            }
            cam_frame = {
                f"observation.images.{camera.config.camera_name}": obs[
                    f"{camera.config.camera_name}_image"
                ]
                for camera in env.cameras
            }
            action_frame = build_dataset_frame(features, action_dict, prefix="action")
            obs_frame = build_dataset_frame(features, obs_dict, prefix="observation.state")

            frame = {**obs_frame, **action_frame, **cam_frame}
            dataset.add_frame(frame, task=args.task)

            obs, _, _, _, _ = env.step(action, block=False)

            sleep_time = 1 / args.fps - (time.time() - step_time_init)
            if sleep_time > 0:
                time.sleep(sleep_time)  # Sleep to allow the environment to process the action

        if recording_manager.state == "paused":
            print(
                "[blue] Stopped episode. Waiting for user to decide whether to save or delete the episode"
            )
            # Start homing for both robots
            leader.home(blocking=False)
            env.robot.home(blocking=False)

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

if args.push_to_hub:
    print(f"[green]Pushing dataset to Hugging Face Hub with repo_id: {args.repo_id}")
    dataset.push_to_hub(repo_id=args.repo_id, private=True)

leader.home()
leader.shutdown()

env.home()
env.close()
