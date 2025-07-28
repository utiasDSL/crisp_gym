"""General manipulator environment configs."""

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from crisp_py.camera.camera_config import CameraConfig
from crisp_py.gripper.gripper import GripperConfig
from crisp_py.robot_config import FrankaConfig, RobotConfig

from crisp_gym.config.path import CRISP_CONFIG_PATH


@dataclass
class ManipulatorEnvConfig(ABC):
    """Manipulator Gym Environment Configuration."""

    control_frequency: float
    robot_config: RobotConfig
    gripper_config: GripperConfig | None
    camera_configs: List[CameraConfig]

    gripper_threshold: float = 0.1
    gripper_enabled: bool = True
    gripper_continuous_control: bool = False

    cartesian_control_param_config: Path | None = field(
        default_factory=lambda: CRISP_CONFIG_PATH / "control" / "default_cartesian_impedance.yaml"
    )
    joint_control_param_config: Path | None = field(
        default_factory=lambda: CRISP_CONFIG_PATH / "control" / "joint_control.yaml"
    )

    max_episode_steps: int | None = None


@dataclass
class FrankaEnvConfig(ManipulatorEnvConfig):
    """Franka Gym Environment Configuration."""

    control_frequency: float = 10.0

    gripper_threshold: float = 0.1

    robot_config: RobotConfig = field(default_factory=lambda: FrankaConfig())

    gripper_config: GripperConfig | None = field(
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
    gripper_continuous_control: bool = True
    gripper_enabled: bool = False

    robot_config: RobotConfig = field(default_factory=lambda: FrankaConfig())

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig(min_value=0, max_value=1)
    )

    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])

    max_episode_steps: int | None = 1000


@dataclass
class LeftNoCamFrankaEnvConfig(NoCamFrankaEnvConfig):
    """Franka Gym Environment Configuration for the left robot without cameras."""

    gripper_enabled: bool = True

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(CRISP_CONFIG_PATH / ("gripper_left.yaml")).resolve()
        )
    )


@dataclass
class RightNoCamFrankaEnvConfig(NoCamFrankaEnvConfig):
    """Franka Gym Environment Configuration for the right robot without cameras."""

    gripper_enabled: bool = True

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(CRISP_CONFIG_PATH / ("gripper_right.yaml")).resolve()
        )
    )


@dataclass
class OnlyWristCamFrankaEnvConfig(ManipulatorEnvConfig):
    """Franka Gym Environment Configuration."""

    control_frequency: float = 10.0

    gripper_threshold: float = 0.1
    gripper_enabled: bool = False

    robot_config: RobotConfig = field(default_factory=lambda: FrankaConfig())

    gripper_config: GripperConfig | None = field(
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
    """Custom Franka Gym Environment Configuration for Franka with an Aloha gripper and cameras."""

    control_frequency: float = 10.0

    # The aloha gripper can be controlled in a continuous manner, so we set this to True.
    gripper_enabled: bool = True
    gripper_continuous_control: bool = True

    robot_config: RobotConfig = field(default_factory=lambda: FrankaConfig())
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(CRISP_CONFIG_PATH / ("gripper_aloha.yaml")).resolve()
        )
    )
    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])

    max_episode_steps: int | None = 1000


@dataclass
class LeftAlohaFrankaEnvConfig(AlohaFrankaEnvConfig):
    """Custom Franka Gym Environment Configuration for the left robot with an Aloha gripper and cameras."""

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(CRISP_CONFIG_PATH / ("gripper_left.yaml")).resolve()
        )
    )

    camera_configs: List[CameraConfig] = field(
        default_factory=lambda: [
            CameraConfig(
                camera_name="primary",
                camera_frame="primary_link",
                resolution=(256, 256),
                camera_color_image_topic="left_third_person_camera/color/image_raw",
                camera_color_info_topic="left_third_person_camera/color/camera_info",
            ),
            CameraConfig(
                camera_name="wrist",
                camera_frame="wrist_link",
                resolution=(256, 256),
                camera_color_image_topic="left_wrist_camera/color/image_rect_raw",
                camera_color_info_topic="left_wrist_camera/color/camera_info",
            ),
        ]
    )


@dataclass
class RightAlohaFrankaEnvConfig(AlohaFrankaEnvConfig):
    """Custom Franka Gym Environment Configuration for the right robot with an Aloha gripper and cameras."""

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(CRISP_CONFIG_PATH / ("gripper_right.yaml")).resolve()
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


@dataclass
class NoCamNoGripperFrankaEnvConfig(ManipulatorEnvConfig):
    """Franka Gym Environment Configuration without cameras and gripper."""

    control_frequency: float = 10.0

    cartesian_control_param_config: Path | None = None
    joint_control_param_config: Path | None = None

    robot_config: RobotConfig = field(default_factory=lambda: FrankaConfig())
    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig(min_value=0, max_value=1)
    )
    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])

    gripper_enabled: bool = False


def make_env_config(
    env_type: str,
    control_frequency: float = 10.0,
) -> ManipulatorEnvConfig:
    """Factory function to create an environment configuration based on the type."""
    config_class = STRING_TO_CONFIG.get(env_type.lower())
    if config_class is None:
        raise ValueError(
            f"Unsupported environment type: {env_type}, available types are {list(STRING_TO_CONFIG.keys())}."
        )
    return config_class(control_frequency=control_frequency)


def list_env_configs() -> list[str]:
    """List all available environment configurations."""
    return list(STRING_TO_CONFIG.keys())


STRING_TO_CONFIG = {
    "right_aloha_franka": RightAlohaFrankaEnvConfig,
    "left_aloha_franka": LeftAlohaFrankaEnvConfig,
    "franka": FrankaEnvConfig,
    "no_cam_franka": NoCamFrankaEnvConfig,
    "left_no_cam_franka": LeftNoCamFrankaEnvConfig,
    "right_no_cam_franka": RightNoCamFrankaEnvConfig,
    "only_wrist_cam_franka": OnlyWristCamFrankaEnvConfig,
    "no_cam_no_gripper_franka": NoCamNoGripperFrankaEnvConfig,
}
