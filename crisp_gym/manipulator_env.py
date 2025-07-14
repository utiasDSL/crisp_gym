"""General manipulator environments."""

import os
from typing import Any, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from crisp_py.camera import Camera
from crisp_py.gripper import Gripper
from crisp_py.robot import Pose, Robot
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from crisp_gym.manipulator_env_config import FrankaEnvConfig, ManipulatorEnvConfig


class ManipulatorBaseEnv(gym.Env):
    """Base class for Manipulator Gym Environment.

    This class serves as a base for creating specific Manipulator Gym environments.
    It cannot be used directly.
    """

    def __init__(self, namespace: str = "", config: Optional[ManipulatorEnvConfig] = None):
        """Initialize the Manipulator Gym Environment.

        Args:
            namespace (str): ROS2 namespace for the robot.
            config (ManipulatorEnvConfig): Configuration for the environment.
        """
        super().__init__()
        self.config = config if config else FrankaEnvConfig()

        if not rclpy.ok():
            rclpy.init()

        self.robot = Robot(
            namespace=namespace,
            robot_config=self.config.robot_config,
        )
        self.gripper = Gripper(
            namespace=namespace,
            gripper_config=self.config.gripper_config,
        )
        self.cameras = [
            Camera(
                namespace=namespace,
                config=camera_config,
            )
            for camera_config in self.config.camera_configs
        ]

        self.timestep = 0
        self.ctrl_type = None

        self.robot.wait_until_ready(timeout=3)
        if self.config.gripper_enabled:
            self.gripper.wait_until_ready(timeout=3)
        for camera in self.cameras:
            camera.wait_until_ready(timeout=3)

        if self.config.cartesian_control_param_config:
            if not os.path.exists(self.config.cartesian_control_param_config):
                raise FileNotFoundError(
                    f"Cartesian control parameter config file not found: {self.config.cartesian_control_param_config}"
                )
            self.robot.cartesian_controller_parameters_client.load_param_config(
                file_path=self.config.cartesian_control_param_config
            )
        if self.config.joint_control_param_config:
            if not os.path.exists(self.config.joint_control_param_config):
                raise FileNotFoundError(
                    f"Joint control parameter config file not found: {self.config.joint_control_param_config}"
                )
            self.robot.joint_controller_parameters_client.load_param_config(
                file_path=self.config.joint_control_param_config
            )

        self.control_rate = self.robot.node.create_rate(self.config.control_frequency)

        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"{camera.config.camera_name}_image": gym.spaces.Box(
                        low=np.zeros((*camera.config.resolution, 3), dtype=np.uint8),
                        high=255 * np.ones((*camera.config.resolution, 3), dtype=np.uint8),
                        dtype=np.uint8,
                    )
                    for camera in self.cameras
                },
                "joint": gym.spaces.Box(
                    low=np.ones((7,), dtype=np.float32) * -np.pi,
                    high=np.ones((7,), dtype=np.float32) * np.pi,
                    dtype=np.float32,
                ),
                "cartesian": gym.spaces.Box(
                    low=-np.ones((6,), dtype=np.float32),
                    high=np.ones((6,), dtype=np.float32),
                    dtype=np.float32,
                ),
                "gripper": gym.spaces.Box(
                    low=np.zeros((1,), dtype=np.float32),
                    high=np.ones((1,), dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

    def _get_obs(self) -> dict:
        """Retrieve the current observation from the robot.

        Returns:
            dict: A dictionary containing the current sensor and state information, including:
                - '{camera_name}_image': RGB image from each configured camera.
                - 'joint': Current joint configuration of the robot in radians.
                - 'cartesian': End-effector pose as a 6D vector (position [xyz] + orientation in Euler angles [xyz], in radians).
                - 'gripper': Normalized gripper state (0 = fully open, 1 = fully closed).
        """
        obs = {}
        for camera in self.cameras:
            obs[f"{camera.config.camera_name}_image"] = camera.current_image
        obs["joint"] = self.robot.joint_values
        obs["cartesian"] = np.concatenate(
            (
                self.robot.end_effector_pose.position,
                self.robot.end_effector_pose.orientation.as_euler("xyz"),
            ),
            axis=0,
        )
        obs["gripper"] = (
            1 - np.array([self.gripper.value]) if self.config.gripper_enabled else np.array([0.0])
        )
        return obs

    def _set_gripper_action(self, action: float):
        """Execute the gripper action.

        Args:
            action (float): Action value for the gripper (0,1).
        """
        if not self.config.gripper_enabled:
            return

        if self.config.gripper_continous_control:
            # If continuous control is enabled, set the gripper value directly
            self.gripper.set_target(action)
        else:
            if action >= self.config.gripper_threshold and self.gripper.is_open(
                open_threshold=self.config.gripper_threshold
            ):
                self.gripper.close()
            elif action < self.config.gripper_threshold and not self.gripper.is_open(
                open_threshold=self.config.gripper_threshold
            ):
                self.gripper.open()

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, bool, dict]:
        """Step the environment.

        Returns truncated flag if max_episode_steps is reached.
        """
        obs = {}
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        if self.config.max_episode_steps and self.timestep >= self.config.max_episode_steps:
            truncated = True

        self.timestep += 1

        return obs, reward, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[dict, dict]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)

        self.timestep = 0

        return self._get_obs(), {}

    def close(self):
        """Close the environment."""
        self.robot.shutdown()
        super().close()

    def switch_controller(self, ctrl_type: str):
        """Switch the controller type.

        Args:
            ctrl_type (str): Type of controller to switch to ('joint' or 'cartesian').
        """
        ctrl_types = {
            "joint": "joint_impedance_controller",
            "cartesian": "cartesian_impedance_controller",
        }

        if ctrl_type not in ctrl_types:
            print(f"Controller {ctrl_type} not availabe.")

            return

        self.ctrl_type = ctrl_type
        self.robot.controller_switcher_client.switch_controller(ctrl_types[self.ctrl_type])

    def home(self, home_config: list[float] | None = None, blocking: bool = True):
        """Move the robot to the home position.

        Args:
            home_config (list[float]): Optional home configuration for the robot.
            blocking (bool): If True, wait until the robot reaches the home position.
        """
        current_ctrl_type = self.ctrl_type
        if current_ctrl_type is None:
            raise ValueError(
                "Control type should be set in the configuration file of the environment."
            )

        if self.config.gripper_enabled:
            self.gripper.open()
        self.robot.home(home_config, blocking)

        if not blocking:
            self.switch_controller(current_ctrl_type)

    def move_to(
        self, position: List | NDArray | None = None, pose: Pose | None = None, speed: float = 0.05
    ):
        """Move the robot to a specified position or pose.

        Args:
            position (iter): Optional position to move to [x, y, z].
            pose (iter): Optional pose (translation and rotation) to move to.
            speed (float): Speed of the movement.
        """
        current_ctrl_type = self.ctrl_type
        if current_ctrl_type is None:
            raise ValueError(
                "Control type should be set in the configuration file of the environment."
            )

        self.switch_controller("cartesian")

        if pose:
            pose = Pose(
                position=np.array(position), orientation=Rotation.from_euler("xyz", np.array(pose))
            )
            position = None

        if self.config.gripper_enabled:
            self.gripper.open()
        self.robot.move_to(position=position, pose=pose, speed=speed)

        self.robot.reset_targets()
        self.robot.wait_until_ready()
        self.switch_controller(current_ctrl_type)


class ManipulatorCartesianEnv(ManipulatorBaseEnv):
    """Manipulator Cartesian Environment.

    This class is a specific implementation of the Manipulator Gym Environment for Cartesian space control.
    """

    def __init__(self, namespace: str = "", config: Optional[ManipulatorEnvConfig] = None):
        """Initialize the Manipulator Cartesian Environment.

        Args:
            namespace (str): ROS2 namespace for the robot.
            config (ManipulatorEnvConfig): Configuration for the environment.
        """
        super().__init__(namespace, config)

        self._min_z_height = 0.0

        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                [
                    -np.ones((3,), dtype=np.float32),
                    -np.ones((3,), dtype=np.float32) * np.pi,
                    np.zeros((1,), dtype=np.float32),
                ],
                axis=0,
            ),
            high=np.concatenate(
                [
                    np.ones((3,), dtype=np.float32),
                    np.ones((3,), dtype=np.float32) * np.pi,
                    np.ones((1,), dtype=np.float32),
                ],
                axis=0,
            ),
            dtype=np.float32,
        )

        self.switch_controller("cartesian")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[dict, dict]:
        """Reset the environment."""
        obs, info = super().reset(seed=seed, options=options)

        self.robot.reset_targets()
        self.robot.wait_until_ready()
        self.switch_controller("cartesian")

        return obs, info

    def step(self, action: np.ndarray, block: bool = True) -> Tuple[dict, float, bool, bool, dict]:
        """Step the environment with a Cartesian action.

        Args:
            action (np.ndarray): Cartesian delta action [dx, dy, dz, roll, pitch, yaw, gripper_action].
            block (bool): If True, block to maintain the control rate.

        Returns:
            Tuple[dict, float, bool, bool, dict]: Observation, reward, terminated flag, truncated flag, and info dictionary.
        """
        assert action.shape == self.action_space.shape, (
            f"Action shape {action.shape} does not match expected shape {self.action_space.shape}"
        )
        # assert self.action_space.contains(action), f"Action {action} is not in the action space {self.action_space}"

        translation, rotation = action[:3], Rotation.from_euler("xyz", action[3:6])

        target_position = self.robot.target_pose.position + translation
        target_position[2] = max(target_position[2], self._min_z_height)
        target_orientation = rotation * self.robot.target_pose.orientation

        target_pose = Pose(position=target_position, orientation=target_orientation)
        self.robot.set_target(pose=target_pose)

        if self.config.gripper_enabled:
            self._set_gripper_action(action[6])

        if block:
            if self.control_rate.time_until_next_call() < 0:
                self.robot.node.get_logger().warn(
                    f"Control rate is not being maintained by {-self.control_rate.time_until_next_call()} seconds."
                )
            self.control_rate.sleep()

        _, reward, terminated, truncated, info = super().step(action)

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info


class ManipulatorJointEnv(ManipulatorBaseEnv):
    """Manipulator Joint Environment.

    This class is a specific implementation of the Manipulator Gym Environment for Joint space control.
    """

    def __init__(self, namespace: str = "", config: Optional[ManipulatorEnvConfig] = None):
        """Initialize the Manipulator Joint Environment.

        Args:
            namespace (str): ROS2 namespace for the robot.
            config (ManipulatorEnvConfig): Configuration for the environment.
        """
        super().__init__(namespace, config)

        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                [np.ones((7,), dtype=np.float32) * -np.pi, np.zeros((1,), dtype=np.float32)], axis=0
            ),
            high=np.concatenate(
                [np.ones((7,), dtype=np.float32) * np.pi, np.ones((1,), dtype=np.float32)], axis=0
            ),
            dtype=np.float32,
        )

        self.switch_controller("joint")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[dict, dict]:
        """Reset the environment."""
        obs, info = super().reset(seed=seed, options=options)

        self.robot.reset_targets()
        self.robot.wait_until_ready()
        self.switch_controller("joint")

        return obs, info

    def step(self, action: np.ndarray, block: bool = True) -> Tuple[dict, float, bool, bool, dict]:
        """Step the environment with a Joint action.

        Args:
            action (np.ndarray): Joint delta action [dtheta1, dtheta2, ..., dtheta7, gripper_action].
            block (bool): If True, block to maintain the control rate.

        Returns:
            Tuple[dict, float, bool, bool, dict]: Observation, reward, terminated flag, truncated flag, and info dictionary.
        """
        obs = {}
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        assert action.shape == self.action_space.shape, (
            f"Action shape {action.shape} does not match expected shape {self.action_space.shape}"
        )
        # assert self.action_space.contains(action), f"Action {action} is not in the action space {self.action_space}"

        # target_joint = (self.robot.target_joint + action[:7] + np.pi) % (2 * np.pi) - np.pi
        target_joint = self.robot.target_joint + action[:7]

        self.robot.set_target_joint(target_joint)

        if self.config.gripper_enabled:
            self._set_gripper_action(action[7])

        if block:
            self.control_rate.sleep()

        _, reward, terminated, truncated, info = super().step(action)

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info
