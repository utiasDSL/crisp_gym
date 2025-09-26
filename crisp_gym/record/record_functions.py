"""Record functions for teleoperation, policy deployment and more in a manipulator environment.

This module should be used in conjunction with the `RecordingManager` class.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.constants import OBS_IMAGES
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import populate_queues

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


def inference_worker(  # noqa: D417
    conn: Connection,
    pretrained_path: str,
    env: ManipulatorBaseEnv,
    steps: int| None,
    inpainting: bool,
    replan_time: int,
):  # noqa: ANN001
    """Policy inference process: loads policy on GPU, receives observations via conn, returns actions, and exits on None.

    Args:
        conn (Connection): The connection to the parent process for sending and receiving data.
        pretrained_path (str): Path to the pretrained policy model.
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
    
    if steps is not None:
        policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
        # Check if the number of steps make sense 
        horizon=policy_config.horizon
        if steps >= horizon: 
            raise ValueError(
            f"The policy steps={steps} must be smaller than the horizon={horizon}. "
            "Please modify your cli."
        )
        obs=policy_config.n_obs_steps
        if steps <= obs: 
            raise ValueError(
            f"The policy must give out steps={steps} bigger than the observation horizon={obs}. "
            "Please modify your cli."
        )
        policy_config.n_action_steps = int(steps)
         # Overwrite to load the new model with the modified config file
        policy = policy_cls.from_pretrained(pretrained_path,config=policy_config)

    else: 
        policy = policy_cls.from_pretrained(pretrained_path)

    logging.info(
        f"[Inference] Loaded {policy.name} policy with {pretrained_path} on device {device}."
    )
    policy.reset()
    policy.to(device).eval()

    # Read policy config to know obs/action window sizes
    cfg = policy.config
    n_obs = int(cfg.n_obs_steps)
    print("Ready to recive information")

    # Set up the policy to consider inpainting. This requires updating the default policies 
    if inpainting:
        # How much to reuse the steps from a previous prediction 
        policy.inpainting = int(replan_time)

    while True:
        # Check if messages are recieved correctly
        msg = conn.recv()
        if msg is None:
            break
        if msg == "reset":
            logging.info("[Inference] Resetting policy")
            policy.reset()
            continue
        if not (isinstance(msg, dict) and msg.get("type") == "OBS_SEQ"):
            logging.warning(f"[Inference] Unknown message: {type(msg)}")
            continue
        
        # We are recieving a list of dictonaries with the last observations 
        obs_seq = msg["obs_seq"]

        # Make the policy predict an action chunk for the current obeservation.
        # Therefore we follow the implementation on the Lerobot side for select_action() which calls predict_action_chunk()
        with torch.inference_mode():
            for i in range(n_obs):
                last= obs_seq[i]
                state = np.concatenate([last["cartesian"][:6], last["gripper"]])
                batch = {
                    "observation.state": torch.from_numpy(state)
                        .unsqueeze(0)
                        .to(device=device, dtype=torch.float32),
                    "task": "", # TODO: Add task description if needed
                }
                for cam in env.cameras:
                    img = last[f"{cam.config.camera_name}_image"]
                    batch[f"observation.images.{cam.config.camera_name}"] = (
                        torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)/ 255
                    )

                # This mirrors Lerobot `select_action()` pre-processing so queues are filled correctly
                batch_norm = policy.normalize_inputs(batch)
                if policy.config.image_features:
                    batch_norm = dict(batch_norm) # shallow copy then add OBS_IMAGES stack
                    batch_norm[OBS_IMAGES] = torch.stack(
                        [batch_norm[k] for k in policy.config.image_features], dim=-4
                    )
                # Note: It's important that this happens after stacking the images into a single key.
                policy._queues = populate_queues(policy._queues, batch_norm)

            # Now get a fresh chunk
            chunk = policy.predict_action_chunk(batch_norm)  
            chunk = chunk.squeeze(0).to(device="cpu").numpy()

        logging.debug(f"[Inference] Computed chunk with shape {tuple(chunk.shape)}")
        conn.send(chunk)

    conn.close()
    logging.info("[Inference] Worker shutting down")

