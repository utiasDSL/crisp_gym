"""Configuration for the teleoperation leader robot."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from crisp_py.gripper import GripperConfig
from crisp_py.robot_config import FrankaConfig

from crisp_gym.config.path import CRISP_CONFIG_PATH

if TYPE_CHECKING:
    from pathlib import Path

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

    disable_gripper_torque: bool = True


@dataclass
class LeftAlohaFrankaTeleopRobotConfig(TeleopRobotConfig):
    """Configuration for the left robot as a leader."""

    leader: RobotConfig = field(default_factory=lambda: FrankaConfig())
    leader_gripper: GripperConfig = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(CRISP_CONFIG_PATH / "trigger_left.yaml").resolve()
        )
    )

    gravity_compensation_controller: Path = field(
        default_factory=lambda: CRISP_CONFIG_PATH / "control" / "gravity_compensation.yaml"
    )

    leader_namespace: str = "left"
    leader_gripper_namespace: str = "left/gripper"


@dataclass
class RightAlohaFrankaTeleopRobotConfig(TeleopRobotConfig):
    """Configuration for the right robot as a leader."""

    leader: RobotConfig = field(default_factory=lambda: FrankaConfig())
    leader_gripper: GripperConfig = field(
        default_factory=lambda: GripperConfig.from_yaml(
            path=(CRISP_CONFIG_PATH / "gripper_right.yaml").resolve()
        )
    )

    gravity_compensation_controller: Path = field(
        default_factory=lambda: CRISP_CONFIG_PATH / "control" / "gravity_compensation.yaml"
    )

    leader_namespace = "right"
    leader_gripper_namespace = "right/gripper"


def make_leader_config(
    name: str,
) -> TeleopRobotConfig:
    """Create a TeleopRobotConfig for the leader robot."""
    if name not in STRING_TO_CONFIG:
        raise ValueError(
            f"Unsupported leader robot type: {name}. Supported types are: {list(STRING_TO_CONFIG.keys())}"
        )

    return STRING_TO_CONFIG[name]()


STRING_TO_CONFIG = {
    "left_aloha_franka": LeftAlohaFrankaTeleopRobotConfig,
    "right_aloha_franka": RightAlohaFrankaTeleopRobotConfig,
}
