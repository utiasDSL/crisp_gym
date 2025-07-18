"""Example on how to teleoperate a robot using another one."""

import time

import numpy as np

from crisp_gym.config.home import home_close_to_table
from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.manipulator_env_config import make_env_config
from crisp_gym.teleop.teleop_robot import TeleopRobot
from crisp_gym.teleop.teleop_robot_config import make_leader_config

try:
    from rich import print
except ImportError:
    pass

USE_FORCE_FEEDBACK = True  # Set to True if you want to use force feedback from the leader robot

# %% Leader setup
print("Setting up leader robot...")
leader_config = make_leader_config("left_aloha_franka")
leader_config.leader.home_config = home_close_to_table
leader_config.leader_gripper.publish_frequency = 30.0
leader_config.leader_gripper.max_delta = 0.15
leader = TeleopRobot(config=leader_config)
leader.wait_until_ready()
leader.gripper.disable_torque()

time.sleep(1.0)  # Wait for the gripper to disable torque

leader.robot.home(blocking=True)

# %% Environment setup
print("Setting up environment...")
env_config = make_env_config("right_no_cam_franka", control_frequency=200.0)
env_config.robot_config.home_config = home_close_to_table
env = ManipulatorCartesianEnv(namespace="right", config=env_config)

env.home()
env.reset()

# %% Now run the teleoperation loop
print("[green bold]Starting teleoperation...")

if USE_FORCE_FEEDBACK:
    leader.robot.controller_switcher_client.switch_controller("torque_feedback_controller")
else:
    leader.robot.cartesian_controller_parameters_client.load_param_config(
        file_path=leader_config.gravity_compensation_controller
    )
    leader.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")


previous_pose = leader.robot.end_effector_pose
while True:
    action_pose = leader.robot.end_effector_pose - previous_pose
    previous_pose = leader.robot.end_effector_pose

    action = np.concatenate(
        [
            action_pose.position,
            action_pose.orientation.as_euler("xyz"),
            [leader.gripper.value],
        ]
    )
    obs, *_ = env.step(action, block=True)
