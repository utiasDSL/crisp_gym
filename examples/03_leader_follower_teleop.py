"""Example on how to teleoperate a robot using another one."""

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


# %% Leader setup
print("Setting up leader robot...")
leader_config = make_leader_config("left_aloha_franka")
leader_config.leader.home_config = home_close_to_table
leader = TeleopRobot(config=leader_config)
leader.wait_until_ready()

leader.robot.home(blocking=True)
leader.robot.cartesian_controller_parameters_client.load_param_config(
    file_path=leader_config.gravity_compensation_controller
)
leader.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
leader.gripper.disable_torque()

# %% Environment setup
print("Setting up environment...")
env_config = make_env_config("right_no_cam_franka", control_frequency=200.0)
env_config.robot_config.home_config = home_close_to_table
env = ManipulatorCartesianEnv(namespace="right", config=env_config)

env.home()
env.reset()

# %% Now run the teleoperation loop
print("[green bold]Starting teleoperation...")

previous_pose = leader.robot.end_effector_pose
while True:
    action_pose = leader.robot.end_effector_pose - previous_pose
    previous_pose = leader.robot.end_effector_pose

    gripper = env.gripper.value + np.clip(
        leader.gripper.value - env.gripper.value,
        -0.25,
        0.25,
    )

    action = np.concatenate(
        [
            action_pose.position,
            action_pose.orientation.as_euler("xyz"),
            [gripper],
        ]
    )
    obs, *_ = env.step(action, block=True)
