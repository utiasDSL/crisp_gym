"""General manipulator environment configs."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from crisp_py.camera.camera_config import CameraConfig
from crisp_py.gripper.gripper import GripperConfig
from crisp_py.robot_config import FrankaConfig, RobotConfig

CRISP_CONFIG_PATH = os.environ.get("CRISP_CONFIG_PATH")
if CRISP_CONFIG_PATH is None:
    raise EnvironmentError(
        "Environment variable 'CRISP_CONFIG_PATH' is not set. Please run:\n"
        "  export CRISP_CONFIG_PATH=/path/to/config"
    )

CRISP_CONFIG_PATH = Path(CRISP_CONFIG_PATH)


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

    gripper_enabled: bool = True
    gripper_continous_control: bool = False

    cartesian_control_param_config: Optional[Path] = field(
        default_factory=lambda: CRISP_CONFIG_PATH / "control" / "default_cartesian_impedance.yaml"
    )
    joint_control_param_config: Optional[Path] = field(
        default_factory=lambda: CRISP_CONFIG_PATH / "control" / "joint_control.yaml"
    )

    max_episode_steps: int | None = None


@dataclass
class FrankaEnvConfig(ManipulatorEnvConfig):
    """Franka Gym Environment Configuration."""

    control_frequency: float = 10.0

    gripper_threshold: float = 0.1

    robot_config: RobotConfig = field(default_factory=lambda: FrankaConfig())

    gripper_config: GripperConfig = field(
        default_factory=lambda: GripperConfig(
            min_value=0,
            max_value=1,
            command_topic="gripper/gripper_position_controller/commands",
            joint_state_topic="gripper/joint_states",
            reboot_service="gripper/reboot_gripper",
            enable_torque_service="gripper/dynamixel_hardware_interface/set_dxl_torque",
        )
    )

    camera_configs: List[CameraConfig] = field(
        default_factory=lambda: [
            CameraConfig(
                camera_name="primary",
                camera_frame="primary_link",
                resolution=(256, 256),
                camera_color_image_topic="right_third_person_camera/color/image_raw",
                camera_color_info_topic="right_third_person_camera/color/camera_info",
            ),
            CameraConfig(
                camera_name="wrist",
                camera_frame="wrist_link",
                resolution=(256, 256),
                camera_color_image_topic="right_wrist_camera/color/image_rect_raw",
                camera_color_info_topic="right_wrist_camera/color/camera_info",
            ),
        ]
    )

    max_episode_steps: int | None = 1000
