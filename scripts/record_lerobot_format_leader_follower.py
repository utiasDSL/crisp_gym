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
from crisp_gym.manipulator_env import ManipulatorCartesianEnv, ManipulatorJointEnv
from crisp_gym.manipulator_env_config import AlohaFrankaEnvConfig
from crisp_gym.record.recording_manager import (
    KeyboardRecordingManager,
    ROSRecordingManager,
)


parser = argparse.ArgumentParser(description="Record data in Lerobot Format")
parser.add_argument(
    "--repo-id",
    type=str,
    default="franka_single",
    help="Repository ID for the dataset",
)
parser.add_argument(
    "--tasks",
    type=str,
    nargs="+",
    default=["pick the lego block."],
    help="List of task descriptions to record data for, e.g. 'clean red' 'clean green'",
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
    default=10,
    help="Number of episodes to record",
)
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume recording of an already existing dataset",
)
parser.add_argument(
    "--push-to-hub",
    action=argparse.BooleanOptionalAction,
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
    "--trigger-config",
    type=str,
    default="trigger",
    help="Trigger configuration for the leader robot.",
)
parser.add_argument(
    "--time-to-home",
    type=float,
    default=3.0,
    help="Time needed to home.",
)
parser.add_argument(
    "--recording-manager-type",
    type=str,
    default="keyboard",
    help="Type of recording manager to use. Currently only 'keyboard' and 'ros' are supported.",
)
parser.add_argument(
    "--leader-side",
    type=str,
    default="left",
    help="Side of the leader robot (left or right).",
)
parser.add_argument(
    "--use-close-home",
    action="store_true",
    help="Whether to use the close home configuration for the robot.",
)

# TODO: @maxdoesch add this
parser.add_argument(
    "--joint-control",
    action="store_true",
    help="Whether to use joint control for the robot.",
)

args = parser.parse_args()

# Clean up existing dataset if it exists
if not args.resume and Path(HF_LEROBOT_HOME / args.repo_id).exists():
    shutil.rmtree(HF_LEROBOT_HOME / args.repo_id)

# TODO: find a fix solution for this
home_config = [
    -1.73960110e-02,
    9.55319758e-02,
    8.09703053e-04,
    -1.94272034e00,
    -4.01435784e-03,
    2.06584183e00,
    7.97426445e-01,
]


# Set up the config for the environment
path_to_config = os.environ.get("CRISP_CONFIG_PATH")
if path_to_config is None:
    raise ValueError(
        "You need to set the environment variable CRISP_CONFIG_PATH in order to load configs for the gripper and controller.\nTo do this execute export CRISP_CONFIG_PATH=path\\to\\config."
    )

if args.leader_side not in ["left", "right"]:
    raise ValueError(
        f"Invalid leader side: {args.leader_side}. It should be either 'left' or 'right'."
    )

if args.leader_side == "right":
    raise NotImplementedError(
        "Currently, only the left side is supported for the leader robot. Please set --leader-side to 'left'."
    )

leader_side = args.leader_side
follower_side = "right" if leader_side == "left" else "left"

# Set up the envionment configuration
gripper_config = GripperConfig.from_yaml(
    path=(Path(path_to_config) / ("gripper_" + follower_side + ".yaml")).resolve()
)
gripper_config.joint_state_topic = "gripper" + "/" + gripper_config.joint_state_topic
gripper_config.command_topic = "gripper" + "/" + gripper_config.command_topic

ctrl_type = "cartesian" if not args.joint_control else "joint"

env_config = AlohaFrankaEnvConfig(gripper_config=gripper_config)
env_config.control_frequency = args.fps
if args.use_close_home:
    env_config.robot_config.home_config = home_config
    env_config.robot_config.time_to_home = args.time_to_home
env = ManipulatorCartesianEnv(namespace=follower_side, config=env_config)

# %% Prepare the Leader
leader = Robot(namespace=leader_side)
if args.use_close_home:
    leader.config.home_config = home_config
    leader.config.time_to_home = args.time_to_home
leader.wait_until_ready()

# NOTE: This is temporary since there is still no trigger in the right side.
if leader_side == "left":
    leader_gripper_path = (Path(path_to_config) / (args.trigger_config + ".yaml")).resolve()
else:
    leader_gripper_path = (Path(path_to_config) / "gripper_right.yaml").resolve()

leader_gripper = Gripper(
    gripper_config=GripperConfig.from_yaml(path=leader_gripper_path),
    namespace=f"{leader_side}/gripper",
)
leader_gripper.wait_until_ready()
leader_gripper.disable_torque()

# %% Prepare the dataset
features = get_features(env, ctrl_type=ctrl_type)

if args.resume:
    print(f"[green]Resuming recording from existing dataset: {args.repo_id}")
    dataset = LeRobotDataset(repo_id=args.repo_id)
    if args.num_episodes <= dataset.num_episodes:
        print(
            f"[yellow] The dataset already has {dataset.num_episodes} recorded. Please select a larger number.[/yellow]"
        )
        exit()
else:
    print(f"[green]Creating new dataset: {args.repo_id}")
    # Clean up existing dataset if it exists
    if Path(HF_LEROBOT_HOME / args.repo_id).exists():
        print(
            "[yellow]WARNING: The repo_id already exists. If you decide to continue, you will overwrite the content. If you intended to resume the collection of data, then execute this script with the --resume flag. Otherwise press <Enter> to continue."
        )
        input()
        shutil.rmtree(HF_LEROBOT_HOME / args.repo_id)
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

tasks = list(args.tasks)

# %% Start interaction
previous_pose = leader.end_effector_pose
previous_joint = leader.joint_values

start_episode_number = dataset.num_episodes

if args.recording_manager_type == "keyboard":
    reconding_manager_cls = KeyboardRecordingManager
elif args.recording_manager_type == "ros":
    reconding_manager_cls = ROSRecordingManager
else:
    raise ValueError(
        f"Unsupported recording manager type: {args.recording_manager_type}. "
        "Currently only 'keyboard' and 'ros' are supported."
    )

with reconding_manager_cls(
    num_episodes=args.num_episodes - start_episode_number
) as recording_manager:
    while not recording_manager == "exit":
        print(
            f"[magenta bold]=== Episode {recording_manager.episode_count + 1 + start_episode_number} / {recording_manager.num_episodes + start_episode_number} ==="
        )
        task = tasks[np.random.randint(0, len(tasks))]
        print(
            f"[magenta bold]=== Task: [italic]{task}[/italic] ==="
        )

        if recording_manager.state == "is_waiting":
            print("[magenta]Waiting for user to start.")
            while recording_manager.state == "is_waiting":
                time.sleep(0.05)

        print("[blue]Started episode")

        start_time = time.time()
        is_first_step = True
        obs = {}

        # HACK: be sure that this one is running. Maybe the controller failed during recording so we need
        # to set the parameters again.
        # Reset before starting
        env.reset()
        leader.reset_targets()
        leader.cartesian_controller_parameters_client.load_param_config(
            file_path=Path(path_to_config) / "control" / (args.leader_controller + ".yaml")
        )
        leader.controller_switcher_client.switch_controller("cartesian_impedance_controller")

        while recording_manager.state == "recording":
            if is_first_step:
                previous_pose = leader.end_effector_pose
                previous_joint = leader.joint_values
                # TODO: @danielsanjosepro make this steps clearer to the user
                obs, _, _, _, _ = env.step(
                    action=np.concatenate(
                        [
                            np.zeros(features["action"]["shape"][0] - 1),
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
                    block=True,
                )

                is_first_step = False
                continue

            # print("Actual time is: ", time.time() - start_time)
            # print("Step time is: ", env.timestep * 1 / 30)

            # TODO: @danielsanjosepro make this steps clearer to the user
            action_pose = leader.end_effector_pose - previous_pose
            action_joint = leader.joint_values - previous_joint
            previous_pose = leader.end_effector_pose
            previous_joint = leader.joint_values

            action = np.concatenate(
                [
                    list(action_pose.position) + list(action_pose.orientation.as_euler("xyz"))
                    if ctrl_type == "cartesian" else list(action_joint),
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
                dim: obs[ctrl_type][i] if i < features["observation.state"]["shape"][0] - 1 else obs["gripper"][0]
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
            dataset.add_frame(frame, task=task)

            obs, _, _, _, _ = env.step(action, block=True)

        if recording_manager.state == "paused":
            print(
                "[blue]Stopped episode. Waiting for user to decide whether to save or delete the episode"
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

if args.not_push_to_hub:
    print(
        "[green]Not pushing dataset to Hugging Face Hub. Use --not-push-to-hub to skip this step."
    )
    # dataset.push_to_hub(repo_id=args.repo_id, private=True)
else:
    print(f"[green]Pushing dataset to Hugging Face Hub with repo_id: {args.repo_id}")
    dataset.push_to_hub(repo_id=args.repo_id, private=True)

leader.home()
leader.shutdown()

env.home()
env.close()