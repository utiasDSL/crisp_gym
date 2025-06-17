from dataclasses import dataclass, field
from typing import List

from crisp_py.robot_config import FrankaConfig, RobotConfig
from crisp_py.camera.camera_config import CameraConfig
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
        default_factory=lambda: GripperConfig(
            min_value=0, 
            max_value=1,
            command_topic = "gripper/gripper_position_controller/commands",
            joint_state_topic = "gripper/joint_states",
            reboot_service = "gripper/reboot_gripper",
            enable_torque_service = "gripper/dynamixel_hardware_interface/set_dxl_torque"
            )
    )

    camera_configs: List[CameraConfig] = field(
        default_factory=lambda: [
            CameraConfig(
                camera_name = "primary",
                camera_frame = "primary_link",
                resolution = (256, 256),
                camera_color_image_topic = "right_third_person_camera/color/image_raw",
                camera_color_info_topic =  "right_third_person_camera/color/camera_info",
            ),
            CameraConfig(
                camera_name = "wrist",
                camera_frame = "wrist_link",
                resolution = (256, 256),
                camera_color_image_topic = "right_wrist_camera/color/image_rect_raw",
                camera_color_info_topic = "right_wrist_camera/color/camera_info",
            ),
        ]
    )

    max_episode_steps: int = 1000