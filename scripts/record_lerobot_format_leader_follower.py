"""Script showcasing how to record data in Lerobot Format."""

import argparse
import logging

import numpy as np
import rclpy  # noqa: F401
from rich.logging import RichHandler

import crisp_gym  # noqa: F401
from crisp_gym.config.home import (
    home_close_to_table,
)
from crisp_gym.config.path import CRISP_CONFIG_PATH
from crisp_gym.manipulator_env import (
    ManipulatorCartesianEnv,
    ManipulatorJointEnv,
)
from crisp_gym.manipulator_env_config import list_env_configs, make_env_config
from crisp_gym.record.record_functions import make_teleop_fn
from crisp_gym.record.recording_manager import (
    KeyboardRecordingManager,
    ROSRecordingManager,
)
from crisp_gym.teleop.teleop_robot import TeleopRobot
from crisp_gym.teleop.teleop_robot_config import list_leader_configs, make_leader_config
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features


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
    default="aloha_franka",
    help="Type of robot being used.",
)
parser.add_argument(
    "--fps",
    type=int,
    default=15,
    help="Frames per second for recording",
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
    "--recording-manager-type",
    type=str,
    default="keyboard",
    help="Type of recording manager to use. Currently only 'keyboard' and 'ros' are supported.",
)
parser.add_argument(
    "--leader-config",
    type=str,
    default=None,
    help="Configuration name for the leader robot. Define your own configuration in `crisp_gym/teleop/teleop_robot_config.py`.",
)
parser.add_argument(
    "--follower-config",
    type=str,
    default=None,
    help="Configuration name for the follower robot. Define your own configuration in `crisp_gym/manipulator_env_config.py`.",
)
parser.add_argument(
    "--follower-namespace",
    type=str,
    default=None,
    help="Namespace for the follower robot. This is used to identify the robot in the ROS ecosystem.",
)
parser.add_argument(
    "--leader-namespace",
    type=str,
    default=None,
    help="Namespace for the leader robot. This is used to identify the robot in the ROS ecosystem.",
)

parser.add_argument(
    "--time-to-home",
    type=float,
    default=3.0,
    help="Time needed to home.",
)
parser.add_argument(
    "--use-close-home",
    action="store_true",
    help="Whether to use the close home configuration for the robot.",
)
parser.add_argument(
    "--joint-control",
    action="store_true",
    help="Whether to use joint control for the robot.",
)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level.",
)

args = parser.parse_args()

FORMAT = "%(message)s"
logging.basicConfig(level=args.log_level, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

# Log the arguments
logging.info("Arguments:")
for arg, value in vars(args).items():
    logging.info(f"  {arg}: {value}")


if args.follower_namespace is None:
    args.follower_namespace = prompt.prompt(
        "Please enter the follower robot namespace (e.g., 'left', 'right', ...)",
        default="right",
    )
    logging.info(f"Using follower namespace: {args.follower_namespace}")

if args.leader_namespace is None:
    args.leader_namespace = prompt.prompt(
        "Please enter the leader robot namespace (e.g., 'left', 'right', ...)",
        default="left",
    )
    logging.info(f"Using leader namespace: {args.leader_namespace}")

if args.leader_config is None:
    leader_configs = list_leader_configs()
    args.leader_config = prompt.prompt(
        "Please enter the leader robot configuration name.",
        options=leader_configs,
        default=leader_configs[0],
    )
    logging.info(f"Using leader configuration: {args.leader_config}")


if args.follower_config is None:
    follower_configs = list_env_configs()
    args.follower_config = prompt.prompt(
        "Please enter the follower robot configuration name.",
        options=follower_configs,
        default=follower_configs[0],
    )
    logging.info(f"Using follower configuration: {args.follower_config}")

try:
    ctrl_type = "cartesian" if not args.joint_control else "joint"

    # Prepare the Follower Environment
    env_config = make_env_config(args.follower_config, control_frequency=args.fps)
    logging.debug(f"Using follower environment configuration: {env_config}")

    if args.use_close_home:
        env_config.robot_config.home_config = home_close_to_table
        env_config.robot_config.time_to_home = args.time_to_home

    env = (
        ManipulatorCartesianEnv(namespace=args.follower_namespace, config=env_config)
        if ctrl_type == "cartesian"
        else ManipulatorJointEnv(namespace=args.follower_namespace, config=env_config)
    )

    # Prepare the Leader
    leader_config = make_leader_config(args.leader_config)
    if args.use_close_home:
        leader_config.leader.home_config = home_close_to_table
        leader_config.leader.time_to_home = args.time_to_home

    leader = TeleopRobot(config=leader_config)
    leader.wait_until_ready()

    # TODO: @danielsanjosepro: add more features to the dataset depending on the user's needs.
    features = get_features(env_config=env_config, ctrl_type=ctrl_type)
    logging.info(features)

    if args.recording_manager_type == "keyboard":
        recording_manager_cls = KeyboardRecordingManager
    elif args.recording_manager_type == "ros":
        recording_manager_cls = ROSRecordingManager
    else:
        raise ValueError(
            f"Unsupported recording manager type: {args.recording_manager_type}. "
            "Currently only 'keyboard' and 'ros' are supported."
        )

    recording_manager = recording_manager_cls(
        features=features,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        num_episodes=args.num_episodes,
        fps=args.fps,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
    )
    recording_manager.wait_until_ready()

    logging.info("Homing both robots before starting with recording.")

    # Prepare environment and leader
    leader.prepare_for_teleop()
    env.home()
    env.reset()

    tasks = list(args.tasks)

    def on_start():
        """Hook function to be called when starting a new episode."""
        env.reset()
        try:
            leader.robot.controller_switcher_client.switch_controller("torque_feedback_controller")
        except Exception:
            leader.robot.cartesian_controller_parameters_client.load_param_config(
                CRISP_CONFIG_PATH / "control" / "gravity_compensation_on_plane.yaml"
            )
            leader.robot.controller_switcher_client.switch_controller(
                "cartesian_impedance_controller"
            )  # TODO: @danielsanjosepro: ask user for which controller to use.

    def on_end():
        """Hook function to be called when stopping the recording."""
        env.robot.reset_targets()
        env.robot.home(blocking=False)
        leader.robot.reset_targets()
        leader.robot.home(blocking=False)

    with recording_manager:
        while not recording_manager.done():
            logging.info(
                f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
            )

            task = tasks[np.random.randint(0, len(tasks))] if tasks else "No task specified."
            logging.info(f"▷ Task: {task}")

            recording_manager.record_episode(
                data_fn=make_teleop_fn(env, leader),
                task=task,
                on_start=on_start,
                on_end=on_end,
            )

    logging.info("Homing leader.")
    leader.robot.home()
    logging.info("Homing follower.")
    env.home()

    logging.info("Closing the environment.")
    env.close()

    logging.info("Finished recording.")

except TimeoutError:
    logging.exception("Timeout error occurred during recording.")
    logging.error(
        "Please check if the robot container is running and the namespace is correct."
        "\nYou can check the topics using `ros2 topic list` command."
    )

except Exception:
    logging.exception("An error occurred during recording.")

finally:
    if rclpy.ok():
        rclpy.shutdown()
