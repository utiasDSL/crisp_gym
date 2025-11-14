"""Record functions for teleoperation, policy deployment and more in a manipulator environment.

This module should be used in conjunction with the `RecordingManager` class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from crisp_gym.util.control_type import ControlType
from crisp_gym.util.gripper_mode import GripperMode

if TYPE_CHECKING:
    from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv, ManipulatorCartesianEnv
    from crisp_gym.teleop.teleop_robot import TeleopRobot
    from crisp_gym.teleop.teleop_sensor_stream import TeleopStreamedPose


logger = logging.getLogger(__name__)


def _leader_gripper_to_action(
    leader_value: float,
    follower_value: float,
    control_mode: GripperMode | str,
) -> float:
    """Convert the leader gripper value to an action for the follower gripper.

    Args:
        leader_value (float): The current value of the leader gripper.
        follower_value (float): The current value of the follower gripper.
        control_mode (GripperMode): The control mode of the gripper.

    Returns:
        float: The computed gripper action for the follower.
    """
    if isinstance(control_mode, str):
        control_mode = GripperMode(control_mode)

    if control_mode in [GripperMode.ABSOLUTE_BINARY, GripperMode.ABSOLUTE_CONTINUOUS]:
        return leader_value
    elif control_mode in [GripperMode.RELATIVE_BINARY, GripperMode.RELATIVE_CONTINUOUS]:
        return leader_value - follower_value
    elif control_mode == GripperMode.NONE:
        return 0.0
    else:
        raise ValueError(f"Unsupported gripper control mode: {control_mode}")


def make_teleop_streamer_fn(env: ManipulatorCartesianEnv, leader: TeleopStreamedPose) -> Callable:
    """Create a teleoperation function for the leader robot using streamed pose data."""
    prev_pose = leader.last_pose
    first_step = True

    def _fn() -> tuple:
        """Teleoperation function to be called in each step.

        This function computes the action based on the current end-effector pose
        or joint values of the leader robot, updates the gripper value, and steps
        the environment.

        Returns:
            tuple: A tuple containing the observation from the environment and the action taken.
        """
        nonlocal prev_pose, first_step
        if first_step:
            first_step = False
            prev_pose = leader.last_pose
            return None, None

        pose = leader.last_pose
        action_pose = pose - prev_pose
        prev_pose = pose

        gripper = leader.gripper.value if leader.gripper is not None else 0.0

        # Use the environment's orientation representation for the rotation part
        rot_action = env.rotation_to_representation(action_pose.orientation)

        action = np.concatenate(
            [
                list(action_pose.position) + list(rot_action),
                [gripper],
            ]
        )
        obs, *_ = env.step(action, block=False)
        return obs, action

    return _fn


def make_teleop_fn(env: ManipulatorBaseEnv, leader: TeleopRobot) -> Callable:
    """Create a teleoperation function for the leader robot.

    This function returns a Callable that can be used to control the leader robot
    in a teleoperation manner. It computes the action based on the difference
    between the current and previous end-effector pose or joint values, and
    updates the gripper value based on the leader gripper's value.

    Args:
        env (ManipulatorBaseEnv): The environment in which the leader robot operates.
        leader (TeleopRobot): The teleoperation leader robot instance.

    Returns:
        Callable: A function that, when called, performs a step in the environment
        and returns the observation and action taken.
    """
    prev_pose = leader.robot.end_effector_pose
    prev_joint = leader.robot.joint_values
    first_step = True

    def _fn() -> tuple:
        """Teleoperation function to be called in each step.

        This function computes the action based on the current end-effector pose
        or joint values of the leader robot, updates the gripper value, and steps
        the environment.

        Returns:
            tuple: A tuple containing the observation from the environment and the action taken.
        """
        nonlocal prev_pose, prev_joint, first_step
        if first_step:
            first_step = False
            prev_pose = leader.robot.end_effector_pose
            prev_joint = leader.robot.joint_values
            return None, None

        pose = leader.robot.end_effector_pose
        joint = leader.robot.joint_values
        action_pose = pose - prev_pose
        action_joint = joint - prev_joint
        prev_pose = pose
        prev_joint = joint

        gripper_action = _leader_gripper_to_action(
            leader_value=leader.gripper.value if leader.gripper is not None else 0.0,
            follower_value=env.gripper.value if env.gripper is not None else 0.0,
            control_mode=env.config.gripper_mode,
        )

        action = None
        if env.ctrl_type is ControlType.CARTESIAN:
            # Use the environment's orientation representation for the rotation part
            rot_action = env.rotation_to_representation(action_pose.orientation)
            action = np.concatenate(
                [list(action_pose.position) + list(rot_action), [gripper_action]]
            )
        elif env.ctrl_type is ControlType.JOINT:
            action = np.concatenate([action_joint, [gripper_action]])
        else:
            raise ValueError(
                f"Unsupported control type: {env.ctrl_type}. "
                "Supported types are 'cartesian' and 'joint' for delta actions."
            )

        obs, *_ = env.step(action, block=False)
        return obs, action

    return _fn
