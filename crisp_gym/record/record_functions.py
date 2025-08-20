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

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

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

        if leader.gripper is None:
            gripper = 0.0
        elif env.gripper is None:
            gripper = leader.gripper.value + np.clip(
                leader.gripper.value - leader.gripper.value,
                -leader.gripper.config.max_delta,
                leader.gripper.config.max_delta,
            )
        else:
            gripper = env.gripper.value + np.clip(
                leader.gripper.value - env.gripper.value,
                -env.gripper.config.max_delta,
                env.gripper.config.max_delta,
            )

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
                "Supported types are 'cartesian' and 'joint'."
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config = TrainPipelineConfig.from_pretrained(pretrained_path)
    if train_config.policy is None:
        raise ValueError(
            f"Policy configuration is missing in the pretrained path: {pretrained_path}. "
            "Please ensure the policy is correctly configured."
        )
    policy_cls = get_policy_class(train_config.policy.type)
    policy = policy_cls.from_pretrained(pretrained_path)

    logging.info(
        f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
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
            logging.exception(f"Error during environment step: {e}")

        return obs_raw, action

    return _fn
