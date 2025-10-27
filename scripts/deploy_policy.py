"""Script showcasing how to record data in Lerobot Format."""

import argparse
import logging
import time
from multiprocessing import Pipe, Process

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import get_policy_class

import crisp_gym  # noqa: F401
from crisp_gym.config.home import home_close_to_table
from crisp_gym.manipulator_env import make_env
from crisp_gym.manipulator_env_config import list_env_configs
from crisp_gym.record.record_functions import inference_worker
from crisp_gym.record.recording_manager import make_recording_manager
from crisp_gym.util import prompt
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach…")
# debugpy.wait_for_client()   
# print("Hello, Debugging!")

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

parser.add_argument(
   "--async-inference",
    type=int,
    default=None,
    help="At which step to start a new prediction during the execution of one chunk. The resulting chunk will be shorter for consitency.",
)

parser.add_argument(
   "--inference-steps",
    type=int,
    default=None,
    help="How many steps should the policy execute from its prediciton",
)

parser.add_argument(
   "--inpainting",
    type=bool,
    default=False,
    help="Wether to use the actions that will be executed while predicting the next action chunk in async mode, as groundtruth in the following denoising steps",
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

if args.async_inference is None: 
    args.async_inference = prompt.prompt(
        "Please enter after which time during the execution of a action chunk a new action chunk should be predicted"
    )
    logging.info(f"Using async inference at: {args.async_inference}")

ctrl_type = "cartesian" if not args.joint_control else "joint"
env = make_env(args.env_config, control_type=ctrl_type, namespace=args.env_namespace)
env.robot.config.home_config= home_close_to_table

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
        "steps": args.inference_steps,
        "inpainting": args.inpainting,
        "replan_time": args.async_inference, 
    },
    daemon=True,
)
inf_proc.start()

time.sleep(1.0)  # Give some time for the process to start

# Get information about the number of actions that should be executed and the number of observations that are required 
# It is important to do this after the inference process has been started. 
# ToDo find out why this is the case 
train_config = TrainPipelineConfig.from_pretrained(args.path)
policy_cls = get_policy_class(train_config.policy.type)
policy = policy_cls.from_pretrained(args.path)
cfg = policy.config
n_obs = int(cfg.n_obs_steps)

if args.inference_steps is not None:
    n_act= args.inference_steps
else:
    n_act = int(cfg.n_action_steps)

# Check if the replan time is correct 
replan_time=args.async_inference 
if replan_time <= 0:
    logging.warning(f"replan_time={replan_time} can not be smaller zero")
    exit(1)
elif replan_time > n_act:
    logging.warning(f"replan_time={replan_time} > n_action_steps={n_act}")
    exit(1)
elif replan_time < n_act // 2:
    logging.warning(f"replan_time={replan_time} < n_action_steps/2={n_act/2} will stall.")
    exit(1)


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
    env.gripper.open()
    env.robot.home(blocking=False)


with recording_manager:
    while not recording_manager.done():
        logging.info(
            f"→ Episode {recording_manager.episode_count + 1} / {recording_manager.num_episodes}"
        )

        recording_manager.record_episode_inference(
            on_start=on_start,
            on_end=on_end,
            env=env,
            conn=parent_conn,
            task="Pick up the lego block.",
            replan_time=replan_time,
            n_obs=n_obs,
            n_act=n_act,
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
