"""General manipulator environment configs."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from crisp_py.camera.camera_config import CameraConfig
from crisp_py.gripper.gripper import GripperConfig
from crisp_py.robot_config import FrankaConfig, RobotConfig

path_to_crisp_py_config = os.environ.get("CRISP_CONFIG_PATH")
if path_to_crisp_py_config is None:
    raise ValueError(
        "You need to set the environment variable CRISP_CONFIG_PATH in order to load configs for the gripper and controller.\nTo do this execute export CRISP_CONFIG_PATH=path\\to\\config."
    )


@dataclass
class ManipulatorEnvConfig:
    """Manipulator Gym Environment Configuration.

    This class is a configuration for the Manipulator Gym Environment.
    It contains the robot and camera configurations.
    """

    control_frequency: float

    robot_config: RobotConfig
    gripper_config: GripperConfig
    camera_configs: List[CameraConfig]

    gripper_enabled: bool = True
    gripper_threshold: float = 0.1
    gripper_continous_control: bool = False

    cartesian_control_param_config: Optional[Path] = field(
        default_factory=lambda: path_to_crisp_py_config / Path("control/cartesian_impedance_controller.yaml")
    )
    joint_control_param_config: Optional[Path] = field(
        default_factory=lambda: path_to_crisp_py_config / Path("control/joint_impedance_controller.yaml")
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


@dataclass
class NoCamFrankaEnvConfig(ManipulatorEnvConfig):
    """Franka Gym Environment Configuration."""

    control_frequency: float = 10.0

    gripper_threshold: float = 0.1
    gripper_enabled: bool = False

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

    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])

    max_episode_steps: int | None = 1000


@dataclass
class OnlyWristCamFrankaEnvConfig(ManipulatorEnvConfig):
    """Franka Gym Environment Configuration."""

    control_frequency: float = 10.0

    gripper_threshold: float = 0.1
    gripper_enabled: bool = False

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
                camera_name="camera",
                camera_frame="wrist_link",
                resolution=(256, 256),
                camera_color_image_topic="camera/wrist_camera/color/image_rect_raw",
                camera_color_info_topic="camera/wrist_camera/color/camera_info",
            ),
        ]
    )

    max_episode_steps: int | None = 1000


@dataclass
class AlohaFrankaEnvConfig(ManipulatorEnvConfig):
    """Custom Franaka Gym Environment Configuration for Franka with an Aloha gripper and cameras."""

    control_frequency: float = 10.0

    # The aloha gripper can be controlled in a continuous manner, so we set this to True.
    gripper_enabled: bool = True
    gripper_continous_control: bool = True

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
