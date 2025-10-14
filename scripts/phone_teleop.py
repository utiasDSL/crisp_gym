"""Example on how to teleoperate a robot using another one."""

import argparse
import logging
import time

import numpy as np

from crisp_gym.envs.manipulator_env import make_env
from crisp_gym.teleop.teleop_sensor_stream import TeleopStreamedPose
from crisp_gym.util.setup_logger import setup_logging

# Parse args:
parser = argparse.ArgumentParser(description="Teleoperation of a leader robot.")
parser.add_argument(
    "--log-level",
    type=str,
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level (default: INFO)",
)
parser.add_argument(
    "--control-frequency",
    type=float,
    default=100.0,
    help="Control frequency in Hz (default: 100.0)",
)


args = parser.parse_args()

# Set up logging
setup_logging(level=args.log_level)
logger = logging.getLogger(__name__)

# %% Leader setup
logger.info("Setting up streamed teleop...")
leader = TeleopStreamedPose()
leader.wait_until_ready()

# %% Environment setup
logger.info("Setting up environment...")
env = make_env("simple_cam", control_type="cartesian", namespace="right")
env.robot.home()
env.reset()

# %% Now run the teleoperation loop
logger.info(":rocket: Starting teleoperation...")


previous_pose = leader.last_pose

while True:
    # NOTE: the leader pose and follower pose will drift apart over time but this is
    #       fine assuming that we are just recording the leader's actions and not absolute positions.

    action_pose = leader.last_pose - previous_pose
    previous_pose = leader.last_pose

    action = np.concatenate(
        [
            action_pose.position,
            action_pose.orientation.as_euler("xyz"),
            np.array([leader.gripper if leader.gripper else 0.0]),
        ]
    )
    logger.debug(f"Leader pose: {leader.last_pose.position}")
    logger.debug(f"Action: {action}")
    obs, *_ = env.step(action, block=False)
    time.sleep(1.0 / args.control_frequency)
