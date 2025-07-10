"""Script showcasing how to record data in Lerobot Format."""

import argparse  # noqa: I001
import os
import time
from pathlib import Path
import PIL.Image  # noqa: F401
import logging

from crisp_py.gripper import Gripper, GripperConfig
from crisp_py.camera import CameraConfig
from crisp_py.robot import Robot
import numpy as np
from rich.logging import RichHandler

import crisp_gym  # noqa: F401
from crisp_gym.lerobot_wrapper import get_features
from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.manipulator_env_config import AlohaFrankaEnvConfig
from crisp_gym.record.recording_manager import (
    KeyboardRecordingManager,
    ROSRecordingManager,
)

FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

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
    # type=bool,
    action="store_true",
    help="Resume recording of an already existing dataset",
)
parser.add_argument(
    "--not-push-to-hub",
    action="store_true",
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
    type=bool,
    default=False,
    help="Whether to use joint control for the robot.",
)

args = parser.parse_args()


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
home_right_leader = [
    -0.02312892,
    -0.10664185,
    -0.0195703,
    -1.75644521,
    -0.00732298,
    1.68992915,
    0.8040582,
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

leader_side = args.leader_side
follower_side = "right" if leader_side == "left" else "left"

# Set up the envionment configuration
gripper_config = GripperConfig.from_yaml(
    path=(Path(path_to_config) / ("gripper_" + follower_side + ".yaml")).resolve()
)
gripper_config.joint_state_topic = "gripper" + "/" + gripper_config.joint_state_topic
gripper_config.command_topic = "gripper" + "/" + gripper_config.command_topic

env_config = AlohaFrankaEnvConfig(gripper_config=gripper_config)
if args.leader_side == "right":
    env_config.camera_configs = [
        CameraConfig(
            camera_name="wrist",
            camera_frame="wrist_link",
            resolution=(256, 256),
            camera_color_image_topic="left_wrist_camera/color/image_rect_raw",
            camera_color_info_topic="left_wrist_camera/color/camera_info",
        ),
    ]
if args.use_close_home:
    env_config.robot_config.home_config = home_config
    env_config.robot_config.time_to_home = args.time_to_home

env = ManipulatorCartesianEnv(namespace=follower_side, config=env_config)
env.wait_until_ready()

# %% Prepare the Leader
leader = Robot(namespace=leader_side)
if args.use_close_home:
    leader.config.home_config = home_right_leader if leader_side == "right" else home_config
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
features = get_features(env)

if args.recording_manager_type == "keyboard":
    reconding_manager_cls = KeyboardRecordingManager
elif args.recording_manager_type == "ros":
    reconding_manager_cls = ROSRecordingManager
else:
    raise ValueError(
        f"Unsupported recording manager type: {args.recording_manager_type}. "
        "Currently only 'keyboard' and 'ros' are supported."
    )

recording_manager = reconding_manager_cls(
    features=features,
    repo_id=args.repo_id,
    task=args.task,
    robot_type=args.robot_type,
    num_episodes=args.num_episodes,
    fps=args.fps,
    resume=args.resume,
)

logging.info("Waiting for dataset writer process to get ready...")
try:
    recording_manager.dataset_ready.wait()
except KeyboardInterrupt:
    logging.error("KeyboardInterrupt received. Exiting...")
    exit()

if not recording_manager.episode_count_queue.empty():
    recording_manager.episode_count = recording_manager.episode_count_queue.get()


logging.info("[green]Dataset writer process is ready.", extra={"markup": True})

logging.info("Homing both robots before starting with recording.")

# %% Prepare environment and leader
env.home()
env.reset()

leader.home()
leader.cartesian_controller_parameters_client.load_param_config(
    file_path=Path(path_to_config) / "control" / (args.leader_controller + ".yaml")
)
leader.controller_switcher_client.switch_controller("cartesian_impedance_controller")
leader_gripper.disable_torque()

start_episode_number = recording_manager.episode_count

# %% Start interaction
previous_pose = leader.end_effector_pose


with recording_manager:
    while not recording_manager == "exit":
        logging.info(
            f"[magenta bold]=== Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes} ===",
            extra={"markup": True},
        )

        if recording_manager.state == "is_waiting":
            logging.info("[magenta]Waiting for user to start.", extra={"markup": True})
            while recording_manager.state == "is_waiting":
                time.sleep(0.05)

        logging.info("[blue]Started episode", extra={"markup": True})

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
        # leader_gripper.disable_torque()

        env.reset()

        while recording_manager.state == "recording":
            step_time_init = time.time()

            if is_first_step:
                previous_pose = leader.end_effector_pose

                sleep_time = 1 / args.fps - (time.time() - step_time_init)
                time.sleep(sleep_time)

                is_first_step = False
                continue
            action_pose = leader.end_effector_pose - previous_pose
            previous_pose = leader.end_effector_pose

            gripper_value = env.gripper.value + np.clip(
                leader_gripper.value - env.gripper.value, -0.5, 0.5
            )

            action = np.concatenate(
                [
                    action_pose.position,
                    action_pose.orientation.as_euler("xyz"),
                    [gripper_value],
                ]
            )

            obs, _, _, _, _ = env.step(action, block=False)

            recording_manager.queue.put({"type": "FRAME", "data": (obs, action)})

            sleep_time = 1 / args.fps - (time.time() - step_time_init)
            if sleep_time > 0:
                time.sleep(sleep_time)  # Sleep to allow the environment to process the action
            else:
                leader.node.get_logger().warning(
                    f"CONTROL LOOP IS TOO SLOW. It took {-sleep_time} longer, current fps: {1.0 / (time.time() - step_time_init)}",
                    throttle_duration_sec=1.0,
                )

        if recording_manager.state == "paused":
            logging.info(
                "[blue]Stopped episode. Waiting for user to decide whether to save or delete the episode",
                extra={"markup": True},
            )
            # Start homing for both robots
            leader.home(blocking=False)
            env.robot.home(blocking=False)

        while recording_manager.state == "paused":
            time.sleep(1.0)

        if recording_manager.state == "to_be_saved":
            recording_manager.save_episode()

        if recording_manager.state == "to_be_deleted":
            recording_manager.delete_episode()

        if recording_manager.state == "exit":
            break

if args.not_push_to_hub:
    logging.info(
        "[green]Not pushing dataset to Hugging Face Hub. Use --not-push-to-hub to skip this step.",
        extra={"markup": True},
    )
else:
    recording_manager.push_to_hub()

recording_manager.shutdown()

logging.info("Homing both robots and closing the environment.")

leader.home()
env.home()

leader.shutdown()
env.close()
