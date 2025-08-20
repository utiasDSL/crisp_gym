"""This module provides a function to generate features for the LeRobotDataset.

It defines the features used by the dataset, including camera images, proprioceptive state,
sensors, and actions based on the environment configuration and control type.
"""

from typing import Dict

try:
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
except ImportError:
    raise ImportError(
        "The 'lerobot' package is required to use this function. "
        "Please use a lerobot environment 'pixi shell -e <rosdistro>-lerobot'."
    )

import logging

from crisp_gym.manipulator_env_config import ManipulatorEnvConfig

logger = logging.getLogger(__name__)


def get_features(
    env_config: ManipulatorEnvConfig,
    ctrl_type: str = "cartesian",
    use_video: bool = True,
) -> Dict[str, Dict]:
    """Get the features used by LeRobotDataset.

    Args:
        env_config (ManipulatorEnvConfig): The environment configuration for the manipulator.
        ctrl_type (str): The type of control used, either "joint" or "cartesian". Defaults to "cartesian".
        use_video (bool): Whether to include video features. Defaults to True.
    """
    if not CODEBASE_VERSION.startswith("v2"):
        logger.warn(
            "Feature generation for LeRobot has been implemented for version 2.x of LeRobotDataset. Expect unexpected behaviour for other versions."
        )

    ctrl_dims = {
        "joint": [f"joint_{idx}" for idx in range(env_config.robot_config.num_joints())]
        + ["gripper"],
        "cartesian": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    }

    if ctrl_type not in ctrl_dims:
        raise ValueError(
            f"Control type {ctrl_type} not supported. Supported types: {list(ctrl_dims.keys())}"
        )

    # Feature configuration
    features = {}

    # Camera features
    camera_features = {
        f"observation.images.{cam.camera_name}": {
            "dtype": "image",
            "shape": (*cam.resolution, 3),
            "names": ["height", "width", "channels"],
        }
        for cam in env_config.camera_configs
        if cam.resolution is not None
    }
    features.update(camera_features)

    if use_video:
        video_features = {
            f"observation.images.{cam.camera_name}": {
                "dtype": "video",
                "shape": (*cam.resolution, 3),
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.fps": env_config.control_frequency,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
            for cam in env_config.camera_configs
            if cam.resolution is not None
        }
        features.update(video_features)

    # Propioceptive
    features["observation.state.joint"] = {
        "dtype": "float32",
        "shape": (len(ctrl_dims["joint"]) - 1,),  # Exclude gripper
        "names": ctrl_dims["joint"][:-1],  # Exclude gripper
    }
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(ctrl_dims["cartesian"]),),
        "names": ctrl_dims["cartesian"],
    }
    features["observation.state.target"] = (
        {
            "dtype": "float32",
            "shape": (len(ctrl_dims["cartesian"]) - 1,),  # Exclude gripper
            "names": ctrl_dims["cartesian"][:-1],  # Exclude gripper
        }
        if ctrl_type == "cartesian"
        else {
            "dtype": "float32",
            "shape": (len(ctrl_dims["joint"]) - 1,),  # Exclude gripper
            "names": ctrl_dims["joint"][:-1],  # Exclude gripper
        }
    )

    # Sensors
    sensor_features = {
        f"observation.state.sensor_{sensor.name}": {
            "dtype": "float32",
            "shape": sensor.shape,
            "names": [f"{sensor.name}_{i}" for i in range(sensor.shape[0])],
        }
        for sensor in env_config.sensor_configs
    }
    features.update(sensor_features)

    # Action
    features["action"] = {
        "dtype": "float32",
        "shape": (len(ctrl_dims[ctrl_type]),),
        "names": ctrl_dims[ctrl_type],
    }

    return features
