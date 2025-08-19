"""Script showcasing how to record data in Lerobot Format."""

import argparse
import logging
import time
from multiprocessing import Pipe, Process

import crisp_gym  # noqa: F401
from crisp_gym.manipulator_env import make_env
from crisp_gym.manipulator_env_config import list_env_configs
from crisp_gym.record.record_functions import inference_worker, make_policy_fn
from crisp_gym.record.recording_manager import make_recording_manager
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
parser.add_argument(
    "--env-config",
    type=str,
    default=None,
    help="Configuration name for the follower robot. Define your own configuration in `crisp_gym/manipulator_env_config.py`.",
)
parser.add_argument(
    "--env-namespace",
    type=str,
    default=None,
    help="Namespace for the follower robot. This is used to identify the robot in the ROS ecosystem.",
)


args = parser.parse_args()
setup_logging(args.log_level)

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


if args.env_namespace is None:
    args.env_namespace = prompt.prompt(
        "Please enter the follower robot namespace (e.g., 'left', 'right', ...)",
        default="right",
    )
    logging.info(f"Using follower namespace: {args.env_namespace}")

if args.env_config is None:
    follower_configs = list_env_configs()
    args.env_config = prompt.prompt(
        "Please enter the follower robot configuration name.",
        options=follower_configs,
        default=follower_configs[0],
    )
    logging.info(f"Using follower configuration: {args.env_config}")


ctrl_type = "cartesian" if not args.joint_control else "joint"
env = make_env(args.env_config, control_type=ctrl_type, namespace=args.env_namespace)

# %% Prepare the dataset
features = get_features(env.config, ctrl_type=ctrl_type)


recording_manager = make_recording_manager(
    recording_manager_type=args.recording_manager_type,
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
