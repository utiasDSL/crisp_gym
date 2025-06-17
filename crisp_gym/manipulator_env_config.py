from dataclasses import dataclass, field
from typing import List

from crisp_py.robot_config import FrankaConfig, RobotConfig
from crisp_py.devices.camera_config import CameraConfig, PrimaryCameraConfig, WristCameraConfig
from crisp_py.gripper.gripper import GripperConfig


@dataclass
class ManipulatorEnvConfig:
    """Manipulator Gym Environment Configuration.

    This class is a configuration for the Manipulator Gym Environment.
    It contains the robot and camera configurations.
    """
    control_frequency: float

    gripper_threshold: float

    robot_config: RobotConfig
    gripper_config: GripperConfig
    camera_configs: List[CameraConfig]

    max_episode_steps: int = None

@dataclass
class FrankaEnvConfig(ManipulatorEnvConfig):
    """Franka Gym Environment Configuration."""
    control_frequency: float = 10.0

    gripper_threshold: float = 0.1

    robot_config: RobotConfig = field(
        default_factory=lambda: FrankaConfig()
    )

    gripper_config: GripperConfig = field(
        default_factory=lambda: GripperConfig(min_value=0, max_value=1)
    )

    camera_configs: List[CameraConfig] = field(
        default_factory=lambda: [
            PrimaryCameraConfig(),
            WristCameraConfig(),
        ]
    )