"""General manipulator environment configs."""

import logging
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml
from crisp_py.camera.camera_config import CameraConfig
from crisp_py.gripper.gripper import GripperConfig
from crisp_py.robot_config import FrankaConfig, RobotConfig, make_robot_config
from crisp_py.sensors.sensor_config import SensorConfig, make_sensor_config

from crisp_gym.config.path import CRISP_CONFIG_PATH, find_config, list_configs_in_folder
from crisp_gym.util.gripper_mode import GripperMode


class ObservationKeys:
    """Standardized keys for observations in manipulator environments."""

    STATE_OBS = "observation.state"

    GRIPPER_OBS = STATE_OBS + ".gripper"
    JOINT_OBS = STATE_OBS + ".joints"
    CARTESIAN_OBS = STATE_OBS + ".cartesian"
    TARGET_OBS = STATE_OBS + ".target"
    SENSOR_OBS = STATE_OBS + ".sensors"

    IMAGE_OBS = "observation.images"


ALLOWED_STATE_OBS_KEYS = {
    ObservationKeys.GRIPPER_OBS,
    ObservationKeys.JOINT_OBS,
    ObservationKeys.CARTESIAN_OBS,
    ObservationKeys.TARGET_OBS,
    ObservationKeys.SENSOR_OBS,
}


@dataclass(kw_only=True)
class ManipulatorEnvConfig(ABC):
    """Manipulator Gym Environment Configuration.

    This class serves as a base configuration for manipulator environments.
    It includes parameters for control frequency, robot configuration,
    gripper configuration, camera configurations, and control parameters.

    Attributes:
        control_frequency (float): The frequency at which the environment is controlled.
        robot_config (RobotConfig): Configuration for the robot.
        gripper_config (GripperConfig | None): Configuration for the gripper, if applicable.
        camera_configs (List[CameraConfig]): List of camera configurations.
        cartesian_control_param_config (Path | None): Path to the Cartesian control parameters configuration file.
        joint_control_param_config (Path | None): Path to the joint control parameters configuration file.
        gripper_threshold (float): Threshold for gripper actions.
        gripper_enabled (bool): Whether the gripper is enabled.
        max_episode_steps (int | None): Maximum number of steps per episode, if applicable.
    """

    control_frequency: float
    robot_config: RobotConfig
    gripper_config: GripperConfig | None
    camera_configs: List[CameraConfig]

    cartesian_control_param_config: Path | None
    joint_control_param_config: Path | None

    sensor_configs: List[SensorConfig] = field(default_factory=lambda: [])

    gripper_mode: GripperMode | str = GripperMode.ABSOLUTE_CONTINUOUS
    gripper_threshold: float = 0.1

    max_episode_steps: int | None = None

    gripper_enabled: bool | None = None  # Deprecated, use gripper_mode instead
    gripper_continuous_control: bool | None = None  # Deprecated, use gripper_mode instead

    def __post_init__(self):
        """Post-initialization checks."""
        if self.gripper_enabled is not None:
            logging.warning(
                "Deprecated: 'gripper_enabled' is deprecated, use 'gripper_mode' instead to set the control mode."
            )
        if self.gripper_continuous_control is not None:
            logging.warning(
                "Deprecated: 'gripper_continuous_control' is deprecated, use 'gripper_mode' instead to set the control mode."
            )

        if isinstance(self.gripper_mode, str):
            self.gripper_mode = GripperMode(self.gripper_mode)

    def get_metadata(self) -> dict:
        """Get metadata about the environment configuration.

        Returns:
            dict: Metadata dictionary containing control frequency, robot type, gripper type, and camera names.
        """
        return {
            "robot_config": self.robot_config.__dict__,
            "gripper_config": self.gripper_config.__dict__ if self.gripper_config else "None",
            "camera_config": [camera.__dict__ for camera in self.camera_configs],
            "sensor_config": [sensor.__dict__ for sensor in self.sensor_configs],
            "gripper_mode": str(self.gripper_mode),
            "gripper_threshold": self.gripper_threshold,
            "cartesian_control_param_config": str(self.cartesian_control_param_config),
            "joint_control_param_config": str(self.joint_control_param_config),
        }

    @classmethod
    def from_yaml(cls, yaml_path: Path, **overrides) -> "ManipulatorEnvConfig":  # noqa: ANN003
        """Load config from YAML file with optional overrides.

        Args:
            yaml_path: Path to the YAML configuration file
            **overrides: Additional parameters to override YAML values

        Returns:
            ManipulatorEnvConfig: Configured environment instance
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Apply overrides
        data.update(overrides)

        # Handle nested configs that need special treatment
        if "robot_config" in data and isinstance(data["robot_config"], dict):
            # Use make_robot_config to handle different robot types
            data["robot_config"] = make_robot_config(**data["robot_config"])

        if "gripper_config" in data and isinstance(data["gripper_config"], dict):
            gripper_cfg = data["gripper_config"]
            if "from_yaml" in gripper_cfg:
                # Load from external YAML file
                gripper_yaml_path = find_config(gripper_cfg["from_yaml"])
                if gripper_yaml_path is None:
                    raise FileNotFoundError(
                        f"Gripper config file '{gripper_cfg['from_yaml']}' not found in any CRISP config paths"
                    )
                data["gripper_config"] = GripperConfig.from_yaml(path=gripper_yaml_path.resolve())
            else:
                data["gripper_config"] = GripperConfig(**gripper_cfg)

        if "camera_configs" in data and isinstance(data["camera_configs"], list):
            data["camera_configs"] = [
                CameraConfig(**cam_cfg) if isinstance(cam_cfg, dict) else cam_cfg
                for cam_cfg in data["camera_configs"]
            ]

        if "sensor_configs" in data and isinstance(data["sensor_configs"], list):
            data["sensor_configs"] = [
                make_sensor_config(**sensor_cfg) if isinstance(sensor_cfg, dict) else sensor_cfg
                for sensor_cfg in data["sensor_configs"]
            ]

        return cls(**data)


# === Franka Robotics FR3 Environment Configurations ===
@dataclass
class FrankaEnvConfig(ManipulatorEnvConfig, ABC):
    """Franka Gym Environment Configuration."""

    control_frequency: float = 30.0

    robot_config: RobotConfig = field(default_factory=lambda: FrankaConfig())

    # Default controller configurations for Franka
    cartesian_control_param_config: Path | None = field(
        default_factory=lambda: find_config("control/default_cartesian_impedance.yaml")
        or CRISP_CONFIG_PATH / "control" / "default_cartesian_impedance.yaml"
    )
    joint_control_param_config: Path | None = field(
        default_factory=lambda: find_config("control/joint_control.yaml")
        or CRISP_CONFIG_PATH / "control" / "joint_control.yaml"
    )


@dataclass
class NoCamFrankaEnvConfig(FrankaEnvConfig):
    """Franka Gym Environment Configuration."""

    gripper_continuous_control: bool = True
    gripper_enabled: bool = False

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig(min_value=0, max_value=1)
    )

    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])


@dataclass
class LeftNoCamFrankaEnvConfig(NoCamFrankaEnvConfig):
    """Franka Gym Environment Configuration for the left robot without cameras."""

    gripper_enabled: bool = True

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(
                find_config("gripper_left.yaml") or CRISP_CONFIG_PATH / "gripper_left.yaml"
            ).resolve()
        )
    )


@dataclass
class RightNoCamFrankaEnvConfig(NoCamFrankaEnvConfig):
    """Franka Gym Environment Configuration for the right robot without cameras."""

    gripper_enabled: bool = True

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(
                find_config("gripper_right.yaml") or CRISP_CONFIG_PATH / "gripper_right.yaml"
            ).resolve()
        )
    )


@dataclass
class OnlyWristCamFrankaEnvConfig(FrankaEnvConfig):
    """Franka Gym Environment Configuration."""

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


@dataclass
class AlohaFrankaEnvConfig(FrankaEnvConfig):
    """Custom Franka Gym Environment Configuration for Franka with an Aloha gripper and cameras."""

    # The aloha gripper can be controlled in a continuous manner, so we set this to True.
    # For more information on the gripper, check: https://github.com/TUM-LSY/aloha4franka
    gripper_enabled: bool = True
    gripper_continuous_control: bool = True

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(
                find_config("gripper_aloha.yaml") or CRISP_CONFIG_PATH / "gripper_aloha.yaml"
            ).resolve()
        )
    )
    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])

    max_episode_steps: int | None = 1000


@dataclass
class LeftAlohaFrankaEnvConfig(AlohaFrankaEnvConfig):
    """Custom Franka Gym Environment Configuration for the left robot with an Aloha gripper and cameras."""

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(
                find_config("gripper_left.yaml") or CRISP_CONFIG_PATH / "gripper_left.yaml"
            ).resolve()
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
            path=(
                find_config("gripper_right.yaml") or CRISP_CONFIG_PATH / "gripper_right.yaml"
            ).resolve()
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
class NoCamNoGripperFrankaEnvConfig(FrankaEnvConfig):
    """Franka Gym Environment Configuration without cameras and gripper."""

    gripper_config: GripperConfig | None = field(
        default_factory=lambda: GripperConfig(min_value=0, max_value=1)
    )
    camera_configs: List[CameraConfig] = field(default_factory=lambda: [])

    gripper_enabled: bool = False


def make_env_config(
    env_type: str,
    config_path: Path | str | None = None,
    **overrides,  # noqa: ANN003
) -> ManipulatorEnvConfig:
    """Factory function to create an environment configuration based on the type.

    This function allows for both predefined environment types and custom YAML configurations.
    It will first check if the type is in the predefined set, and if not, it will look for a YAML config file.

    Args:
        env_type: Type of environment configuration
        config_path: Optional path to YAML config file
        **overrides: Additional parameters to override defaults/YAML values

    Returns:
        ManipulatorEnvConfig: Configured environment instance
    """
    config_class = STRING_TO_CONFIG.get(env_type.lower())
    if config_class is None:
        # Try to find YAML config if not in predefined types
        config_path = find_config("envs/" + env_type.lower() + ".yaml")
        if config_path is None:
            raise ValueError(
                f"Unsupported environment type: {env_type}. The list of supported types are: {list_env_configs()}"
            )
        config_class = ManipulatorEnvConfig

    if config_path:
        config_path = Path(config_path) if isinstance(config_path, str) else config_path
        return config_class.from_yaml(config_path, **overrides)

    return config_class(**overrides)


def list_env_configs() -> list[str]:
    """List all available environment configurations."""
    predefined = list(STRING_TO_CONFIG.keys())
    other = list_configs_in_folder("envs")
    yaml_configs = [file.stem for file in other if file.suffix == ".yaml"]
    return predefined + yaml_configs


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
