"""Script showcasing how to record data in Lerobot Format."""

import argparse
import logging
import time
from multiprocessing import Pipe, Process

from lerobot.configs.train import TrainPipelineConfig
from rich.logging import RichHandler

import crisp_gym  # noqa: F401
from crisp_gym.config.home import (
    home_close_to_table,
)
from crisp_gym.lerobot_wrapper import get_features
from crisp_gym.manipulator_env import (
    ManipulatorCartesianEnv,
    ManipulatorJointEnv,
)
from crisp_gym.manipulator_env_config import make_env_config
from crisp_gym.record.record_functions import inference_worker, make_policy_fn
from crisp_gym.record.recording_manager import (
    KeyboardRecordingManager,
    ROSRecordingManager,
)
from crisp_gym.util import prompt

parser = argparse.ArgumentParser(description="Record data in Lerobot Format")
parser.add_argument(
    "--repo-id",
    type=str,
    default="franka_single",
    help="Repository ID for the dataset",
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
parser.add_argument(
    "--path",
    type=str,
    default=None,
    help="Path to save the recordings.",
)

args = parser.parse_args()

FORMAT = "%(message)s"
logging.basicConfig(level=args.log_level, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logging.info("-" * 40)
logging.info("Arguments:")
for arg, value in vars(args).items():
    logging.info(f"  {arg}: {value}")
logging.info("-" * 40)


if args.path is None:
    logging.info(" No path provided. Searching for models in 'outputs/train' directory.")
    from pathlib import Path

    # We check recursively in the 'outputs/train' directory for 'pretrained_model's recursively
    models_path = Path("outputs/train")
    if models_path.exists() and models_path.is_dir():
        models = [model for model in models_path.glob("**/pretrained_model") if model.is_dir()]
        models_names = sorted([str(model) for model in models], key=lambda x: x.lower())

        args.path = prompt.prompt(
            message="Please select a model to use for deployment:",
            options=models_names,
            default=models_names[0] if models else None,
        )
        logging.info(f"Using model path: {args.path}")
    else:
        logging.error("'outputs/models' directory does not exist.")
        logging.error("Please provide a valid path to the model using --path or create a new one.")
        exit(1)


# Set up the config for the environment
follower_side = "right"
ctrl_type = "cartesian" if not args.joint_control else "joint"

# %% Prepare the Follower Environment
env_config = make_env_config(f"{follower_side}_{args.robot_type}", control_frequency=args.fps)
if args.use_close_home:
    env_config.robot_config.home_config = home_close_to_table
    env_config.robot_config.time_to_home = args.time_to_home

env = (
    ManipulatorCartesianEnv(namespace=follower_side, config=env_config)
    if ctrl_type == "cartesian"
    else ManipulatorJointEnv(namespace=follower_side, config=env_config)
)

# %% Prepare the dataset
features = get_features(env, ctrl_type=ctrl_type)


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
    robot_type=args.robot_type,
    num_episodes=args.num_episodes,
    fps=args.fps,
    resume=args.resume,
)
recording_manager.wait_until_ready()

# %% Set up multiprocessing for policy inference
logging.info("Setting up multiprocessing for policy inference.")
parent_conn, child_conn = Pipe()


# Start inference process
inf_proc = Process(
    target=inference_worker,
    kwargs={
        "conn": child_conn,
        "pretrained_path": args.path,
        "env": env,
    },
    daemon=True,
)
inf_proc.start()

time.sleep(1.0)  # Give some time for the process to start

logging.info("Homing robot before starting with recording.")

env.home()
env.reset()


def on_start():
    """Hook function to be called when starting a new episode."""
    env.reset()
    parent_conn.send("reset")


def on_end():
    """Hook function to be called when stopping the recording."""
    env.robot.reset_targets()
    env.robot.home(blocking=False)


with recording_manager:
    while not recording_manager.done():
        logging.info(
            f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
        )

        recording_manager.record_episode(
            data_fn=make_policy_fn(env, parent_conn),
            task="Pick up the lego block.",
            on_start=on_start,
            on_end=on_end,
        )

        logging.info("Episode finished. Waiting for the next episode to start.")

# Shutdown inference process
logging.info("Shutting down inference process.")
parent_conn.send(None)
inf_proc.join()

logging.info("Homing robot.")
env.home()

logging.info("Closing the environment.")
env.close()

logging.info("Finished recording.")
