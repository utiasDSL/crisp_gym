"""Script showcasing how to record data in Lerobot Format."""

import argparse
import logging
import multiprocessing
import time
from multiprocessing import Pipe, Process, connection
from multiprocessing.connection import Connection
from typing import Callable  # noqa: I001

import numpy as np
import torch
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

import crisp_gym  # noqa: F401
from crisp_gym.config.home import (
    home_close_to_table,
)
from crisp_gym.lerobot_wrapper import get_features
from crisp_gym.manipulator_env import (
    ManipulatorBaseEnv,
    ManipulatorCartesianEnv,
    ManipulatorJointEnv,
)
from crisp_gym.manipulator_env_config import make_env_config
from crisp_gym.record.recording_manager import (
    KeyboardRecordingManager,
    ROSRecordingManager,
)

from rich.logging import RichHandler


def inference_worker(conn: Connection, policy_path: str, env: ManipulatorBaseEnv):  # noqa: ANN001
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DiffusionPolicy.from_pretrained(policy_path)
    logging.info(
        f"[Inference] Loaded policy from {policy_path} on device {device} with {policy.config.num_inference_steps} inference steps."
    )
    policy.reset()
    policy.to(device).eval()

    warmup_obs = {"observation.state": torch.zeros((1, 7), device=device), "task": ""}
    for cam in env.cameras:
        if cam.config.resolution is None:
            raise ValueError(
                f"Camera {cam.config.camera_name} does not have a resolution set. "
                "Please set the resolution in the camera configuration."
            )
        warmup_obs[f"observation.images.{cam.config.camera_name}"] = torch.zeros(
            (1, 3, *cam.config.resolution), device=device, dtype=torch.float32
        )

    with torch.inference_mode():
        _ = policy.select_action(warmup_obs)
        torch.cuda.synchronize()

    logging.info("[Inference] Warm-up complete")

    while True:
        obs_raw = conn.recv()
        if obs_raw is None:
            break
        if obs_raw == "reset":
            logging.info("[Inference] Resetting policy")
            policy.reset()
            continue

        with torch.inference_mode():
            state = np.concatenate([obs_raw["cartesian"][:6], obs_raw["gripper"]])
            obs = {
                "observation.state": torch.from_numpy(state).unsqueeze(0).cuda().float(),
                "task": "",  # TODO: Add task description if needed
            }
            for cam in env.cameras:
                img = obs_raw[f"{cam.config.camera_name}_image"]
                obs[f"observation.images.{cam.config.camera_name}"] = (
                    torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
                )

            action = policy.select_action(obs)

        logging.debug(f"[Inference] Computed action: {action}")
        conn.send(action)

    conn.close()
    logging.info("[Inference] Worker shutting down")


def make_policy_fn(env: ManipulatorBaseEnv, parent_conn: Connection) -> Callable:
    """Create a function to apply a policy in the environment using multiprocessing.

    This function returns a Callable that, when invoked, observes the current state
    of the environment, sends the observation to the inference worker via pipe,
    receives the action, and steps the environment with that action.

    Args:
        env (ManipulatorBaseEnv): The environment in which the policy will be applied.
        parent_conn (Connection): The connection to the inference worker for sending observations

    Returns:
        Callable: A function that, when called, performs a step in the environment
        using the policy and returns the observation and action taken.
    """

    def _fn() -> tuple:
        """Function to apply the policy in the environment.

        This function observes the current state of the environment, sends the observation
        to the inference worker, receives the action, and steps the environment.

        Returns:
            tuple: A tuple containing the observation from the environment and the action taken.
        """
        obs_raw = env._get_obs()

        # Send observation to inference worker and receive action
        parent_conn.send(obs_raw)
        action = parent_conn.recv().squeeze(0).to("cpu").numpy()
        logging.debug(f"Action: {action}")

        try:
            env.step(action, block=False)
        except Exception as e:
            logging.error(f"Error during environment step: {e}")

        return obs_raw, action

    return _fn


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
    default=20,
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

args = parser.parse_args()

FORMAT = "%(message)s"
logging.basicConfig(level=args.log_level, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logging.info("Arguments:")
for arg, value in vars(args).items():
    logging.info(f"  {arg}: {value}")

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

# %% Set up multiprocessing for policy inference
logging.info("Setting up multiprocessing for policy inference.")
parent_conn, child_conn = Pipe()

# Start inference process
inf_proc = Process(
    target=inference_worker,
    args=(child_conn, "models/pretrained_model", env),
    daemon=True,
)
inf_proc.start()

time.sleep(1.0)  # Give some time for the process to start

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
    env.home(blocking=True)


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
