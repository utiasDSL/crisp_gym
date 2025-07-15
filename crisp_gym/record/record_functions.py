"""Record functions for teleoperation, policy deployment and more in a manipulator environment.

This module should be used in conjunction with the `RecordingManager` class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from crisp_gym.manipulator_env import ManipulatorBaseEnv
    from crisp_gym.teleop.teleop_robot import TeleopRobot


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

        gripper = env.gripper.value + np.clip(
            leader.gripper.value - env.gripper.value,
            -env.gripper.config.max_delta,
            env.gripper.config.max_delta,
        )

        action = np.concatenate(
            [
                list(action_pose.position) + list(action_pose.orientation.as_euler("xyz"))
                if env.ctrl_type == "cartesian"
                else list(action_joint),
                [gripper],
            ]
        )
        obs, *_ = env.step(action, block=False)
        return obs, action

    return _fn


def make_policy_fn(env: ManipulatorBaseEnv, policy: Callable) -> Callable:
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
        obs = env._get_obs()
        action = policy(obs)
        env.step(action)
        return obs, action

    return _fn
