"""General manipulator environments.

To use an environment, you can use the `make_env` function to create an instance of the desired environment type.

Example:
```python
from crisp_gym import make_env

env = make_env(
    env_type="manipulator_cartesian",
    control_type="cartesian",
    namespace="robot_namespace",
)

env.wait_until_ready()  # Wait for the environment to be ready i.e. received all information from the robot, gripper, cameras, and sensors.

obs, info = env.reset()  # Reset the environment to the initial state
while True:
    action = policy.sample_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```


"""

import logging
import os
from pathlib import Path
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
import rclpy
from crisp_py.camera import Camera
from crisp_py.gripper import Gripper
from crisp_py.robot import Pose, Robot
from crisp_py.sensors.sensor import make_sensor
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from typing_extensions import override

from crisp_gym.manipulator_env_config import ManipulatorEnvConfig, make_env_config
from crisp_gym.util.control_type import ControlType

logger = logging.getLogger(__name__)


class ManipulatorBaseEnv(gym.Env):
    """Base class for Manipulator Gym Environment.

    This class serves as a base for creating specific Manipulator Gym environments.
    It cannot be used directly.
    """

    def __init__(self, config: ManipulatorEnvConfig, namespace: str = ""):
        """Initialize the Manipulator Gym Environment.

        Args:
            namespace (str): ROS2 namespace for the robot.
            config (ManipulatorEnvConfig): Configuration for the environment.
        """
        super().__init__()
        self.config = config

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
        self.sensors: List = [
            make_sensor(
                namespace=namespace,
                sensor_config=sensor_config,
            )
            for sensor_config in self.config.sensor_configs
        ]
        for sensor in self.sensors:
            logger.debug(f"Sensor topic: {sensor.config.data_topic}")

        self.timestep = 0
        self.ctrl_type = ControlType.UNDEFINED

        logger.debug(f"ManipulatorBaseEnv initialized with config: {self.config}")

        self.control_rate = self.robot.node.create_rate(self.config.control_frequency)

        if any(camera.config.resolution is None for camera in self.cameras):
            raise ValueError(
                "All cameras must have a resolution defined in the configuration file."
            )

        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"observation.images.{camera.config.camera_name}": gym.spaces.Box(
                        low=np.zeros((*camera.config.resolution, 3), dtype=np.uint8),
                        high=255 * np.ones((*camera.config.resolution, 3), dtype=np.uint8),
                        dtype=np.uint8,
                    )
                    for camera in self.cameras
                    if camera.config.resolution is not None
                },
                # Combined state: cartesian pose (6D)
                "observation.state.cartesian": gym.spaces.Box(
                    low=np.concatenate(
                        [
                            -np.ones((6,), dtype=np.float32),  # cartesian pose
                        ]
                    ),
                    high=np.concatenate(
                        [
                            np.ones((6,), dtype=np.float32),  # cartesian pose
                        ]
                    ),
                    dtype=np.float32,
                ),
                # Gripper state
                "observation.state.gripper": gym.spaces.Box(
                    low=np.array([0.0], dtype=np.float32),  # Gripper closed
                    high=np.array([1.0], dtype=np.float32),  # Gripper open
                    dtype=np.float32,
                ),
                # Joint state
                "observation.state.joint": gym.spaces.Box(
                    low=np.ones((self.config.robot_config.num_joints(),), dtype=np.float32)
                    * -np.pi,
                    high=np.ones((self.config.robot_config.num_joints(),), dtype=np.float32)
                    * np.pi,
                    dtype=np.float32,
                ),
                # Task description
                "task": gym.spaces.Text(max_length=256),
                # Sensor data
                **{
                    f"observation.state.sensor_{sensor.config.name}": gym.spaces.Box(
                        low=-np.inf * np.ones(sensor.config.shape, dtype=np.float32),
                        high=np.inf * np.ones(sensor.config.shape, dtype=np.float32),
                        dtype=np.float32,
                    )
                    for sensor in self.sensors
                    if hasattr(sensor, "config") and hasattr(sensor.config, "shape")
                },
            }
        )
        self._uninitialized = True

    def initialize(self, force: bool = False):
        """Initialize the environment.

        Args:
            force (bool): If True, force re-initialization even if already initialized.

        Raises:
            FileNotFoundError: If the cartesian or joint control parameter config file does not exist.
        """
        if not self._uninitialized and not force:
            logger.debug("Environment is already initialized.")
            return

        self.wait_until_ready()

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

    def wait_until_ready(self):
        """Wait until the robot, gripper, cameras, and sensors are ready."""
        logger.debug("Waiting for robot, gripper, cameras, and sensors to be ready...")

        self.robot.wait_until_ready(timeout=3)

        if self.config.gripper_enabled:
            self.gripper.wait_until_ready(timeout=3)

        for camera in self.cameras:
            camera.wait_until_ready(timeout=3)

        for sensor in self.sensors:
            sensor.wait_until_ready(timeout=3)

        logger.debug("*Robot, gripper, cameras, and sensors are ready.*")

    def get_obs(self) -> dict:
        """Retrieve the current observation from the robot in LeRobot format and allow backward compatibility."""
        return self._get_obs()

    def _get_obs(self) -> dict:
        """Retrieve the current observation from the robot in LeRobot format.

        Returns:
            dict: A dictionary containing the current sensor and state information in LeRobot format:
                - 'observation.images.{camera_name}': RGB image from each configured camera.
                - 'observation.state': Combined state vector (cartesian pose + gripper).
                - 'observation.state.joint': Current joint configuration of the robot in radians.
                - 'observation.state.sensor_{sensor_name}': Sensor values.
                - 'task': Task description (empty string for now).
        """
        obs = {}

        # TODO: Task description
        obs["task"] = ""

        # TODO: consider using a different representation for rotation that is not Euler angles -> axis-angle or quaternion representation
        cartesian_pose = np.concatenate(
            (
                self.robot.end_effector_pose.position,
                self.robot.end_effector_pose.orientation.as_euler("xyz"),
            ),
            axis=0,
        )
        gripper_value = (
            1 - np.array([self.gripper.value]) if self.config.gripper_enabled else np.array([0.0])
        )

        # Cartesian pose
        obs["observation.state.cartesian"] = cartesian_pose.astype(np.float32)

        # Gripper state
        obs["observation.state.gripper"] = gripper_value.astype(np.float32)

        # Joint state
        obs["observation.state.joint"] = self.robot.joint_values

        # Camera images
        for camera in self.cameras:
            obs[f"observation.images.{camera.config.camera_name}"] = camera.current_image

        # Sensor data
        for sensor in self.sensors:
            obs[f"observation.state.sensor_{sensor.config.name}"] = sensor.value

        return obs

    def _set_gripper_action(self, action: float):
        """Execute the gripper action.

        Args:
            action (float): Action value for the gripper (0,1).
        """
        if not self.config.gripper_enabled:
            return

        if self.config.gripper_continuous_control:
            # If continuous control is enabled, set the gripper value directly
            self.gripper.set_target(action)
        elif self.config.gripper_absolute:
            if action < self.config.gripper_threshold and self.gripper.is_open(
                open_threshold=self.config.gripper_threshold
            ):
                self.gripper.close()
            elif action >= self.config.gripper_threshold and not self.gripper.is_open(
                open_threshold=self.config.gripper_threshold
            ):
                self.gripper.open()
        else:
            if action < 0 and self.gripper.is_open(
                open_threshold=self.config.gripper_threshold
            ):
                self.gripper.close()
            elif action > 0 and not self.gripper.is_open(
                open_threshold=self.config.gripper_threshold
            ):
                self.gripper.open()

    @override
    def step(self, action: np.ndarray, block: bool = False) -> Tuple[dict, float, bool, bool, dict]:
        """Step the environment.

        Args:
            action (np.ndarray): Action to be executed in the environment.
            block (bool): If True, block to maintain the control rate.
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

    @override
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[dict, dict]:
        """Reset the environment."""
        super().reset(seed=seed, options=options)

        self.timestep = 0

        self.initialize()

        self.robot.reset_targets()
        self.robot.wait_until_ready()
        self.switch_to_default_controller()

        return self._get_obs(), {}

    @override
    def close(self):
        """Close the environment."""
        self.robot.shutdown()
        super().close()

    def switch_controller(self, control_type: str | ControlType):
        """Switch the controller type.

        This method switches the controller type of the robot to the specified control type.

        Args:
            control_type (str | ControlType): The control type to switch to.
        """
        desired_ctrl_type = (
            control_type
            if isinstance(control_type, ControlType)
            else ControlType.from_string(control_type)
        )

        self.robot.controller_switcher_client.switch_controller(desired_ctrl_type.controller_name())

    def switch_to_default_controller(self):
        """Switch to the default controller type."""
        self.switch_controller(self.ctrl_type)

    def home(self, home_config: list[float] | None = None, blocking: bool = True):
        """Move the robot to the home position.

        Args:
            home_config (list[float]): Optional home configuration for the robot.
            blocking (bool): If True, wait until the robot reaches the home position.
        """
        if self.config.gripper_enabled:
            self.gripper.open()
        self.robot.home(home_config=home_config, blocking=blocking)

        if not blocking:
            self.switch_to_default_controller()

    def move_to(
        self, position: List | NDArray | None = None, pose: List | NDArray | None = None, speed: float = 0.05
    ):
        """Move the robot to a specified position or pose.

        Args:
            position (iter): Optional position to move to [x, y, z].
            pose (iter): Optional pose (rotation in euler angles xyz) to move to.
            speed (float): Speed of the movement.
        """
        if self.ctrl_type is ControlType.UNDEFINED:
            raise ValueError(
                "Control type should be set in the configuration file of the environment."
            )

        self.switch_controller(ControlType.CARTESIAN)

        if pose is not None:
            pose = Pose(
                position=np.array(position), orientation=Rotation.from_euler("xyz", np.array(pose))
            )
            position = None

        if self.config.gripper_enabled:
            self.gripper.open()
        self.robot.move_to(position=position, pose=pose, speed=speed)

        self.robot.reset_targets()
        self.robot.wait_until_ready()
        self.switch_to_default_controller()


class ManipulatorCartesianEnv(ManipulatorBaseEnv):
    """Manipulator Cartesian Environment.

    This class is a specific implementation of the Manipulator Gym Environment for Cartesian space control.
    """

    def __init__(self, config: ManipulatorEnvConfig, namespace: str = ""):
        """Initialize the Manipulator Cartesian Environment.

        Args:
            namespace (str): ROS2 namespace for the robot.
            config (ManipulatorEnvConfig): Configuration for the environment.
        """
        super().__init__(namespace=namespace, config=config)

        self.ctrl_type = ControlType.CARTESIAN

        # TODO: Make this configurable
        self._min_z_height = 0.0

        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                **self.observation_space.spaces,
                "observation.state.target": gym.spaces.Box(
                    low=np.ones((6,), dtype=np.float32) * -np.pi,
                    high=np.ones((6,), dtype=np.float32) * np.pi,
                    dtype=np.float32,
                ),
            },
        )
        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                [
                    -np.ones((3,), dtype=np.float32),  # Translation limits [-1, -1, -1]
                    -np.ones((3,), dtype=np.float32) * np.pi,  # Rotation limits [-pi, -pi, -pi]
                    -np.ones((1,), dtype=np.float32), # Gripper relative action: -1 = close
                    if self.config.gripper_enabled and not self.config.gripper_config.absolute_actions else np.zeros((1,), dtype=np.float32)  # Gripper absolute action (0 = close)
                ],
                axis=0,
            ),
            high=np.concatenate(
                [
                    np.ones((3,), dtype=np.float32),  # Translation limits [1, 1, 1]
                    np.ones((3,), dtype=np.float32) * np.pi,  # Rotation limits [pi, pi, pi]
                    np.ones((1,), dtype=np.float32),  # Gripper action (1 = open)
                ],
                axis=0,
            ),
            dtype=np.float32,
        )

    @override
    def _get_obs(self) -> dict:
        obs = super()._get_obs()
        # TODO: consider using a different representation for rotation that is not Euler angles -> axis-angle or quaternion representation
        obs["observation.state.target"] = np.concatenate(
            (
                self.robot.target_pose.position,
                self.robot.target_pose.orientation.as_euler("xyz"),
            ),
            axis=0,
        )
        return obs

    @override
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
            self._set_gripper_action(action[-1])

        if block:
            self.control_rate.sleep()

        _, reward, terminated, truncated, info = super().step(action)

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info


class ManipulatorJointEnv(ManipulatorBaseEnv):
    """Manipulator Joint Environment.

    This class is a specific implementation of the Manipulator Gym Environment for Joint space control.
    """

    def __init__(self, config: ManipulatorEnvConfig, namespace: str = ""):
        """Initialize the Manipulator Joint Environment.

        Args:
            namespace (str): ROS2 namespace for the robot.
            config (ManipulatorEnvConfig): Configuration for the environment.
        """
        super().__init__(config=config, namespace=namespace)

        self.ctrl_type = ControlType.JOINT

        self.num_joints = self.config.robot_config.num_joints()

        # We add the target to the observation space to allow the agent to learn the target joint positions.
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict(
            {
                **self.observation_space.spaces,
                "observation.state.target": gym.spaces.Box(
                    low=np.ones((self.num_joints,), dtype=np.float32) * -np.pi,
                    high=np.ones((self.num_joints,), dtype=np.float32) * np.pi,
                    dtype=np.float32,
                ),
            },
        )

        self.action_space = gym.spaces.Box(
            low=np.concatenate(
                [
                    np.ones((self.num_joints,), dtype=np.float32) * -np.pi,  # Joint limits
                    np.zeros((1,), dtype=np.float32),  # Gripper action (0 = close)
                ],
                axis=0,
            ),
            high=np.concatenate(
                [
                    np.ones((self.num_joints,), dtype=np.float32) * np.pi,  # Joint limits
                    np.ones((1,), dtype=np.float32),  # Gripper action (1 = open)
                ],
                axis=0,
            ),
            dtype=np.float32,
        )

    @override
    def _get_obs(self) -> dict:
        obs = super()._get_obs()
        obs["observation.state.target"] = self.robot.target_joint
        return obs

    @override
    def step(self, action: np.ndarray, block: bool = True) -> Tuple[dict, float, bool, bool, dict]:
        """Step the environment with a Joint action.

        Args:
            action (np.ndarray): Joint delta action [dtheta_1, dtheta_2, ..., dtheta_n, gripper_action].
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
        # target_joint = (self.robot.target_joint + action[:, self.num_joints] + np.pi) % (2 * np.pi) - np.pi
        target_joint = self.robot.target_joint + action[: self.num_joints]

        self.robot.set_target_joint(target_joint)

        if self.config.gripper_enabled:
            self._set_gripper_action(action[-1])

        if block:
            self.control_rate.sleep()

        _, reward, terminated, truncated, info = super().step(action)

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info


def make_env(
    env_type: str,
    control_type: str = "cartesian",
    namespace: str = "",
    config_path: Path | str | None = None,
    **config_overrides,  # noqa: ANN003
) -> ManipulatorBaseEnv:
    """Create a manipulator environment instance using the specified configuration.

    Args:
        env_type (str): The name of the environment configuration to use.
        control_type (str): The control type ("cartesian" or "joint"). Defaults to "cartesian".
        namespace (str): Namespace for the robot. Defaults to "".
        config_path (str | None): Optional path to YAML config file.
        **config_overrides: Additional parameters to override configuration defaults.

    Returns:
        ManipulatorBaseEnv: A fully initialized manipulator environment instance.

    Raises:
        ValueError: If the specified environment type or control type is not supported.
    """
    config = make_env_config(env_type, config_path=config_path, **config_overrides)

    if control_type.lower() == "cartesian":
        return ManipulatorCartesianEnv(config=config, namespace=namespace)
    elif control_type.lower() == "joint":
        return ManipulatorJointEnv(config=config, namespace=namespace)
    else:
        raise ValueError(
            f"Unsupported control type: {control_type}. Supported types are: 'cartesian', 'joint'"
        )
