"""Configuration for the teleoperation leader robot."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from crisp_py.gripper import GripperConfig
from crisp_py.robot import FrankaConfig, make_robot_config

from crisp_gym.config.path import CRISP_CONFIG_PATH, find_config, list_configs_in_folder

if TYPE_CHECKING:
    from crisp_py.robot import RobotConfig


@dataclass
class TeleopRobotConfig(ABC):
    """Configuration for the teleoperation leader robot.

    Attributes:
        leader (RobotConfig): Configuration for the leader robot.
        leader_gripper (GripperConfig): Configuration for the gripper of the leader robot.
        gravity_compensation_controller (Path): Path to the gravity compensation controller configuration.
        leader_namespace (str): Namespace for the leader robot.
        leader_gripper_namespace (str): Namespace for the leader robot's gripper.

    """

    leader: RobotConfig
    leader_gripper: GripperConfig

    gravity_compensation_controller: Path

    leader_namespace: str = ""
    leader_gripper_namespace: str = ""

    use_gripper: bool = True
    disable_gripper_torque: bool = True
    use_force_feedback: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: Path, **overrides) -> "TeleopRobotConfig":  # noqa: ANN003
        """Load config from YAML file with optional overrides.

        Args:
            yaml_path: Path to the YAML configuration file
            **overrides: Additional parameters to override YAML values

        Returns:
            TeleopRobotConfig: Configured teleop robot instance
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        data.update(overrides)

        if "gravity_compensation_controller" in data:
            data["gravity_compensation_controller"] = find_config(
                data["gravity_compensation_controller"]
            )

        # Handle nested configs that need special treatment
        if "leader" in data and isinstance(data["leader"], dict):
            data["leader"] = make_robot_config(**data["leader"])

        if "leader_gripper" in data and isinstance(data["leader_gripper"], dict):
            gripper_cfg = data["leader_gripper"]
            if "from_yaml" in gripper_cfg:
                gripper_yaml_path = find_config(gripper_cfg["from_yaml"])
                data["leader_gripper"] = GripperConfig.from_yaml(path=gripper_yaml_path.resolve())
            else:
                data["leader_gripper"] = GripperConfig(**gripper_cfg)

        return cls(**data)


@dataclass
class LeftAlohaFrankaTeleopRobotConfig(TeleopRobotConfig):
    """Configuration for the left robot as a leader."""

    leader: RobotConfig = field(default_factory=lambda: FrankaConfig())
    leader_gripper: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(find_config("trigger_left.yaml")).resolve()
        )
    )

    gravity_compensation_controller: Path = field(
        default_factory=lambda: find_config("control/gravity_compensation.yaml")
    )

    leader_namespace: str = "left"
    leader_gripper_namespace: str = "left/trigger"


@dataclass
class RightAlohaFrankaTeleopRobotConfig(TeleopRobotConfig):
    """Configuration for the right robot as a leader."""

    leader: RobotConfig = field(default_factory=lambda: FrankaConfig())
    leader_gripper: GripperConfig | None = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(find_config("gripper_right.yaml")).resolve()
        )
    )

    gravity_compensation_controller: Path = field(
        default_factory=lambda: find_config("control/gravity_compensation.yaml")
    )

    leader_namespace = "right"
    leader_gripper_namespace = "right/gripper"


@dataclass
class NoGripperTeleopRobotConfig(TeleopRobotConfig):
    """Configuration for a teleoperation robot without a gripper."""

    leader: RobotConfig = field(default_factory=lambda: FrankaConfig())
    leader_gripper: GripperConfig | None = None

    gravity_compensation_controller: Path = field(
        default_factory=lambda: find_config("control/gravity_compensation.yaml")
        or CRISP_CONFIG_PATH / "control" / "gravity_compensation.yaml"
    )


@dataclass
class RightNoGripperTeleopRobotConfig(NoGripperTeleopRobotConfig):
    """Configuration for a teleoperation robot without a gripper."""

    leader_namespace: str = "right"


@dataclass
class LeftNoGripperTeleopRobotConfig(NoGripperTeleopRobotConfig):
    """Configuration for a teleoperation robot without a gripper."""

    leader_namespace: str = "left"


def make_leader_config(
    name: str,
    config_path: Path | str | None = None,
    **overrides,  # noqa: ANN003
) -> TeleopRobotConfig:
    """Factory function to create a teleop robot configuration based on the type.

    This function allows for both predefined teleop robot types and custom YAML configurations.
    It will first check if the type is in the predefined set, and if not, it will look for a YAML config file.

    Args:
        name: Type of teleop robot configuration
        config_path: Optional path to YAML config file
        **overrides: Additional parameters to override defaults/YAML values

    Returns:
        TeleopRobotConfig: Configured teleop robot instance
    """
    config_class = STRING_TO_CONFIG.get(name.lower())
    if config_class is None:
        # Try to find YAML config if not in predefined types
        config_path = find_config("teleop/" + name.lower() + ".yaml")
        if config_path is None:
            raise ValueError(
                f"Unsupported leader robot type: {name}. The list of supported types are: {list_leader_configs()}"
            )
        config_class = TeleopRobotConfig

    if config_path:
        config_path = Path(config_path) if isinstance(config_path, str) else config_path
        return config_class.from_yaml(config_path, **overrides)

    return config_class(**overrides)


def list_leader_configs() -> list[str]:
    """List all available leader robot configurations."""
    predefined = list(STRING_TO_CONFIG.keys())
    other = list_configs_in_folder("teleop")
    yaml_configs = [file.stem for file in other if file.suffix == ".yaml"]
    return predefined + yaml_configs


STRING_TO_CONFIG = {
    "left_aloha_franka": LeftAlohaFrankaTeleopRobotConfig,
    "right_aloha_franka": RightAlohaFrankaTeleopRobotConfig,
    "no_gripper": NoGripperTeleopRobotConfig,
    "right_no_gripper": RightNoGripperTeleopRobotConfig,
    "left_no_gripper": LeftNoGripperTeleopRobotConfig,
}
