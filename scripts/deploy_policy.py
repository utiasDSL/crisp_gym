"""Script showcasing how to record data in Lerobot Format."""

import argparse
import logging
from typing import Callable  # noqa: I001

import numpy as np
import torch
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.pretrained import PreTrainedPolicy

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


def make_policy_fn(env: ManipulatorBaseEnv, policy: PreTrainedPolicy) -> Callable:
    """Create a function to apply a policy in the environment.

    This function returns a Callable that, when invoked, observes the current state
    of the environment, applies the given policy to determine the action, and steps
    the environment with that action.

    Args:
        env (ManipulatorBaseEnv): The environment in which the policy will be applied.
        policy (Callable): A Callable that takes an observation and returns an action.

    Returns:
        Callable: A function that, when called, performs a step in the environment
        using the policy and returns the observation and action taken.
    """

    # WARNING: this is an example of a policy function that can be used with the recording manager. But is untested.
    def _fn() -> tuple:
        """Function to apply the policy in the environment.

        This function observes the current state of the environment, applies the policy,
        and steps the environment with the action returned by the policy.

        Returns:
            tuple: A tuple containing the observation from the environment and the action taken.
        """
        obs_raw = env._get_obs()
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
            obs = {}

            # Propioceptive state
            state = np.concatenate([obs_raw["cartesian"][:6], obs_raw["gripper"]])
            obs = {"observation.state": torch.from_numpy(state).unsqueeze(0).cuda().float()}

            for cam in env.cameras:
                img = obs_raw[f"{cam.config.camera_name}_image"]
                obs[f"observation.images.{cam.config.camera_name}"] = (
                    torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda().float() / 255
                )

            obs["task"] = ""

            # Compute the next action with the policy
            # based on the current observation
            action = policy.select_action(obs)

            # Remove batch dimension
            action = action.squeeze(0)

            # Move to cpu, if not already the case
            action = action.to("cpu")

        return obs_raw, action

    return _fn


try:
    from rich.logging import RichHandler

    FORMAT = "%(message)s"
    logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])
except ImportError:
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level="INFO", format=FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

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

args = parser.parse_args()

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

# %% Prepare the policy
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device '{device}' for the policy inference.")
policy = DiffusionPolicy.from_pretrained(
    repo_id="lerobot/diffusion_policy_franka_single",
    device=device,
)
policy.reset()
policy.to(device).eval()

# %% Warm up CUDA kernels
warmup_obs = {"observation.state": torch.zeros((1, 7), device=device)}
for cam in env.cameras:
    warmup_obs[f"observation.images.{cam.config.camera_name}"] = torch.zeros(
        (1, 3, *cam.config.resolution), device=device, dtype=torch.float32
    )
warmup_obs["task"] = "Pick up the lego block."

with torch.inference_mode():
    _ = policy.select_action(warmup_obs)
    torch.cuda.synchronize()

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
            data_fn=make_policy_fn(env),
            task="Pick up the lego block.",
            on_start=on_start,
            on_end=on_end,
        )


logging.info("Homing robot.")
env.home()

logging.info("Closing the environment.")
env.close()

logging.info("Finished recording.")
