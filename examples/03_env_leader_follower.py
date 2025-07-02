"""Example on how to teleoperate a robot using another one."""

import time
from pathlib import Path

import numpy as np
from crisp_py.gripper import Gripper, GripperConfig
from crisp_py.robot import Robot

from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.manipulator_env_config import NoCamFrankaEnvConfig

env_config = NoCamFrankaEnvConfig()
env_config.gripper_enabled = True

path = Path("../crisp_py/config/gripper_right.yaml").resolve()
env_config.gripper_config = GripperConfig.from_yaml(path=path)
env_config.gripper_config.joint_state_topic = "gripper/gripper_state_broadcaster/joint_states"
env_config.gripper_config.command_topic = "gripper/gripper_position_controller/commands"
env_config.gripper_continous_control = True

env = ManipulatorCartesianEnv(namespace="right", config=env_config)

# %% Leader
leader = Robot(namespace="left")
leader.wait_until_ready()

path = Path("../crisp_py/config/trigger.yaml").resolve()
leader_gripper = Gripper(
    gripper_config=GripperConfig.from_yaml(path=path),
    namespace="left/gripper",
    index=1,
)
leader_gripper.wait_until_ready()
leader_gripper.value

# %% Prepare environment and leader

env.home()
env.reset()

env.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")

leader.wait_until_ready()
leader.cartesian_controller_parameters_client.load_param_config(
    file_path="../crisp_py/config/control/gravity_compensation.yaml"
)
leader.home()
leader.controller_switcher_client.switch_controller("cartesian_impedance_controller")

# %% Start interaction

start_time = time.time()
duration = 30  # seconds

previous_pose = leader.end_effector_pose
while time.time() - start_time < duration:
    action_pose = leader.end_effector_pose - previous_pose
    previous_pose = leader.end_effector_pose

    action = np.concatenate(
        [
            action_pose.position,
            action_pose.orientation.as_euler("xyz"),
            np.array(
                [
                    min(
                        1.0,
                        max(0.0, leader_gripper.value if leader_gripper.value is not None else 0.0),
                    )
                ]
            ),
        ]
    )
    obs, _, _, _, _ = env.step(action, block=False)

    time.sleep(1.0 / 200.0)  # Sleep to allow the environment to process the action

    # TODO: Save action and observation


# %%
