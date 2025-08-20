"""Script showcasing how to record data in Lerobot Format."""

import argparse
import logging

import numpy as np
import rclpy  # noqa: F401

import crisp_gym  # noqa: F401
from crisp_gym.config.path import CRISP_CONFIG_PATH
from crisp_gym.manipulator_env import make_env
from crisp_gym.manipulator_env_config import list_env_configs
from crisp_gym.record.record_functions import make_teleop_fn
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.teleop.teleop_robot import make_leader
from crisp_gym.teleop.teleop_robot_config import list_leader_configs
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging

parser = argparse.ArgumentParser(description="Record data in Lerobot Format")
parser.add_argument(
    "--repo-id",
    type=str,
    default="test",
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
    "--joint-control",
    action="store_true",
    help="Whether to use joint control for the robot.",
)
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logger level.",
)

args = parser.parse_args()

# Set up logger
logger = logging.getLogger(__name__)
setup_logging(level=args.log_level)


logger.info("Arguments:")
for arg, value in vars(args).items():
    logger.info(f"  {arg}: {value}")


# Validate arguments not passed by the user
if args.follower_namespace is None:
    args.follower_namespace = prompt.prompt(
        "Please enter the follower robot namespace (e.g., 'left', 'right', ...)",
        default="right",
    )
    logger.info(f"Using follower namespace: {args.follower_namespace}")

if args.leader_namespace is None:
    args.leader_namespace = prompt.prompt(
        "Please enter the leader robot namespace (e.g., 'left', 'right', ...)",
        default="left",
    )
    logger.info(f"Using leader namespace: {args.leader_namespace}")

if args.leader_config is None:
    leader_configs = list_leader_configs()
    args.leader_config = prompt.prompt(
        "Please enter the leader robot configuration name.",
        options=leader_configs,
        default=leader_configs[0],
    )
    logger.info(f"Using leader configuration: {args.leader_config}")


if args.follower_config is None:
    follower_configs = list_env_configs()
    args.follower_config = prompt.prompt(
        "Please enter the follower robot configuration name.",
        options=follower_configs,
        default=follower_configs[0],
    )
    logger.info(f"Using follower configuration: {args.follower_config}")


try:
    ctrl_type = "cartesian" if not args.joint_control else "joint"

    env = make_env(
        env_type=args.follower_config,
        ctrl_type=ctrl_type,
        namespace=args.follower_namespace,
    )

    leader = make_leader(args.leader_config, namespace=args.leader_namespace)
    leader.wait_until_ready()

    features = get_features(env_config=env.config, ctrl_type=ctrl_type)
    logger.debug(f"Using the features: {features}")

    recording_manager = make_recording_manager(
        recording_manager_type=args.recording_manager_type,
        features=features,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        num_episodes=args.num_episodes,
        fps=args.fps,
        resume=args.resume,
        push_to_hub=args.push_to_hub,
    )
    recording_manager.wait_until_ready()

    logger.info("Homing both robots before starting with recording.")

    # Prepare environment and leader
    leader.prepare_for_teleop()
    env.home()
    env.reset()

    tasks = list(args.tasks)

    def on_start():
        """Hook function to be called when starting a new episode."""
        env.reset()

        # TODO: @danielsanjosepro: ask user for which controller to use.
        try:
            leader.robot.controller_switcher_client.switch_controller("torque_feedback_controller")
        except Exception:
            leader.robot.cartesian_controller_parameters_client.load_param_config(
                CRISP_CONFIG_PATH / "control" / "gravity_compensation_on_plane.yaml"
            )
            leader.robot.controller_switcher_client.switch_controller(
                "cartesian_impedance_controller"
            )

    def on_end():
        """Hook function to be called when stopping the recording."""
        env.robot.reset_targets()
        env.robot.home(blocking=False)
        leader.robot.reset_targets()
        leader.robot.home(blocking=False)

    with recording_manager:
        while not recording_manager.done():
            logger.info(
                f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
            )

            task = tasks[np.random.randint(0, len(tasks))] if tasks else "No task specified."
            logger.info(f"▷ Task: {task}")

            recording_manager.record_episode(
                data_fn=make_teleop_fn(env, leader),
                task=task,
                on_start=on_start,
                on_end=on_end,
            )

    logger.info("Homing leader.")
    leader.robot.home()
    logger.info("Homing follower.")
    env.home()

    logger.info("Closing the environment.")
    env.close()

    logger.info("Finished recording.")

except TimeoutError as e:
    logger.exception(f"Timeout error occurred during recording: {e}.")
    logger.error(
        "Please check if the robot container is running and the namespace is correct."
        "\nYou can check the topics using `ros2 topic list` command."
    )

except Exception as e:
    logger.exception(f"An error occurred during recording: {e}.")

finally:
    if rclpy.ok():
        rclpy.shutdown()
