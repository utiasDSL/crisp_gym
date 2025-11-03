"""Record functions for teleoperation, policy deployment and more in a manipulator environment.

This module should be used in conjunction with the `RecordingManager` class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import get_policy_class

from crisp_gym.util.control_type import ControlType
from crisp_gym.util.lerobot_features import concatenate_state_features, numpy_obs_to_torch
from crisp_gym.util.setup_logger import setup_logging

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from crisp_gym.envs.manipulator_env import ManipulatorBaseEnv, ManipulatorCartesianEnv
    from crisp_gym.teleop.teleop_robot import TeleopRobot
    from crisp_gym.teleop.teleop_sensor_stream import TeleopStreamedPose


logger = logging.getLogger(__name__)


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

        action = np.concatenate(
            [
                list(action_pose.position) + list(action_pose.orientation.as_euler("xyz")),
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

        gripper = leader.gripper.value if leader.gripper is not None else 0.0

        action = None
        if env.ctrl_type is ControlType.CARTESIAN:
            action = np.concatenate(
                [
                    list(action_pose.position) + list(action_pose.orientation.as_euler("xyz")),
                    [gripper],
                ]
            )
        elif env.ctrl_type is ControlType.JOINT:
            action = np.concatenate(
                [
                    action_joint,
                    [gripper],
                ]
            )
        else:
            raise ValueError(
                f"Unsupported control type: {env.ctrl_type}. "
                "Supported types are 'cartesian' and 'joint' for delta actions."
            )

        obs, *_ = env.step(action, block=False)
        return obs, action

    return _fn


def inference_worker(
    conn: Connection,
    pretrained_path: str,
    env: ManipulatorBaseEnv,
):  # noqa: ANN001
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
        dataset_metadata (LeRobotDatasetMetadata): Metadata for the dataset, if needed.
        env (ManipulatorBaseEnv): The environment in which the policy will be applied.
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("[Inference] Starting inference worker...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Inference] Using device: {device}")

        logger.info(f"[Inference] Loading training config from {pretrained_path}...")
        train_config = TrainPipelineConfig.from_pretrained(pretrained_path)
        logger.info("[Inference] Loaded training config.")

        if train_config.policy is None:
            raise ValueError(
                f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
                "Please ensure the policy is correctly configured."
            )

        logger.info("[Inference] Loading policy...")
        policy_cls = get_policy_class(train_config.policy.type)
        policy = policy_cls.from_pretrained(pretrained_path)

        logger.info(
            f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
        )
        policy.reset()
        policy.to(device).eval()

        warmup_obs_raw = env.observation_space.sample()
        warmup_obs_raw["observation.state"] = concatenate_state_features(warmup_obs_raw)
        warmup_obs = numpy_obs_to_torch(warmup_obs_raw)

        logger.info("[Inference] Warming up policy...")
        elapsed_list = []
        with torch.inference_mode():
            import time

            for _ in range(100):
                start = time.time()
                _ = policy.select_action(warmup_obs)
                end = time.time()
                elapsed = end - start
                elapsed_list.append(elapsed)

            torch.cuda.synchronize()

        avg_elapsed = sum(elapsed_list) / len(elapsed_list)
        std_elapsed = np.std(elapsed_list)
        max_elapsed = max(elapsed_list)
        min_elapsed = min(elapsed_list)
        logger.info(
            f"[Inference] Warm-up timing over 100 runs: "
            f"avg={avg_elapsed * 1000:.2f}ms, std={std_elapsed * 1000:.2f}ms, max={max_elapsed * 1000:.2f}ms, min={min_elapsed * 1000:.2f}ms"
        )

        logger.info("[Inference] Warm-up complete")

        while True:
            obs_raw = conn.recv()
            if obs_raw is None:
                break
            if obs_raw == "reset":
                logger.info("[Inference] Resetting policy")
                policy.reset()
                continue

            with torch.inference_mode():
                obs = numpy_obs_to_torch(obs_raw)
                action = policy.select_action(obs)

            logger.debug(f"[Inference] Computed action: {action}")
            conn.send(action)
    except Exception as e:
        logger.exception(f"[Inference] Exception in inference worker: {e}")

    conn.close()
    logger.info("[Inference] Worker shutting down")


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
        logger.debug("Requesting action from policy...")
        obs_raw = env.get_obs()

        obs_raw["observation.state"] = concatenate_state_features(obs_raw)

        parent_conn.send(obs_raw)
        action = parent_conn.recv().squeeze(0).to("cpu").numpy()
        logger.debug(f"Action: {action}")

        try:
            env.step(action, block=False)
        except Exception as e:
            logger.exception(f"Error during environment step: {e}")

        return obs_raw, action

    return _fn
