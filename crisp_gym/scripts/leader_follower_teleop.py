"""Example on how to teleoperate a robot using another one."""

import argparse
import logging
import time

import numpy as np

from crisp_gym.manipulator_env import make_env
from crisp_gym.teleop.teleop_robot import make_leader
from crisp_gym.util.setup_logger import setup_logging

# Parse args:
parser = argparse.ArgumentParser(description="Teleoperation of a leader robot.")
parser.add_argument(
    "--use-force-feedback",
    action="store_true",
    help="Use force feedback from the leader robot (default: False)",
)
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
logger.info("Setting up leader robot...")
leader = make_leader(name="left_aloha_franka", namespace="left")
leader.wait_until_ready()
leader.prepare_for_teleop()

# %% Environment setup
logger.info("Setting up environment...")
env = make_env("right_no_cam_franka", control_type="cartesian", namespace="right")
env.robot.home()
env.reset()

# %% Now run the teleoperation loop
logger.info(":rocket: Starting teleoperation...")

if args.use_force_feedback:
    leader.robot.controller_switcher_client.switch_controller("torque_feedback_controller")
else:
    leader.robot.cartesian_controller_parameters_client.load_param_config(
        file_path=leader.config.gravity_compensation_controller
    )
    leader.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")


previous_pose = leader.robot.end_effector_pose

while True:
    # NOTE: the leader pose and follower pose will drift apart over time but this is
    #       fine assuming that we are just recording the leader's actions and not absolute positions.

    action_pose = leader.robot.end_effector_pose - previous_pose
    previous_pose = leader.robot.end_effector_pose

    action = np.concatenate(
        [
            action_pose.position,
            action_pose.orientation.as_euler("xyz"),
            np.array([leader.gripper.value if leader.gripper else 0.0]),
        ]
    )
    obs, *_ = env.step(action, block=False)
    time.sleep(1.0 / args.control_frequency)
