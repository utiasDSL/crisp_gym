"""This module provides functions to generate and convert features for the LeRobotDataset.

It defines the features used by the dataset, including camera images, proprioceptive state,
sensors, and actions based on the environment configuration and control type. It also provides
utilities for converting observations between different formats.
"""

from typing import Any, Dict

import numpy as np
import rich

try:
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
except ImportError:
    raise ImportError(
        "The 'lerobot' package is required to use this function. "
        "Please use a lerobot environment 'pixi shell -e <rosdistro>-lerobot'."
    )

import logging

from crisp_gym.manipulator_env_config import ManipulatorEnvConfig, make_env_config

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
        "shape": (len(ctrl_dims["cartesian"]) - 1,),  # Exclude gripper
        "names": ctrl_dims["cartesian"][:-1],  # Exclude gripper
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


def convert_observation_to_features(
    obs: Dict[str, Any], features: Dict[str, Dict]
) -> Dict[str, Any]:
    """Convert raw observation dictionary to feature-formatted observations.

    This function takes raw observations from the environment and converts them to the
    standardized LeRobot feature format. It handles mapping from legacy observation keys
    to the new feature-based keys and ensures proper data types.

    Args:
        obs (Dict[str, Any]): Raw observation dictionary from the environment
        features (Dict[str, Dict]): Feature configuration dictionary

    Returns:
        Dict[str, Any]: Converted observations matching the feature format

    Raises:
        ValueError: If required features are missing from observations
    """
    converted_obs = {}

    for feature_name, feature_config in features.items():
        if feature_name == "action":
            # Actions are handled separately
            continue

        if feature_name in obs:
            # Direct feature match
            value = obs[feature_name]
            if isinstance(value, np.ndarray) and feature_name.startswith("observation.state"):
                converted_obs[feature_name] = value.astype(np.float32)
            else:
                converted_obs[feature_name] = value

    return converted_obs


def validate_features_match_observation(obs: Dict[str, Any], features: Dict[str, Dict]) -> bool:
    """Validate that observations contain all required features.

    Args:
        obs (Dict[str, Any]): Observation dictionary to validate
        features (Dict[str, Dict]): Required features

    Returns:
        bool: True if all required features can be satisfied, False otherwise
    """
    missing_features = []

    for feature_name in features:
        if feature_name == "action":
            continue  # Actions are handled separately

        if feature_name in obs:
            continue

        if feature_name.startswith("observation.images."):
            camera_name = feature_name.split(".")[-1]
            image_key = f"{camera_name}_image"
            if image_key not in obs:
                missing_features.append(f"{feature_name} (expected key: {image_key})")

        elif feature_name == "observation.state":
            if not ("cartesian" in obs and "gripper" in obs):
                missing_features.append(f"{feature_name} (missing cartesian or gripper)")

        elif feature_name == "observation.state.joint":
            if "joint" not in obs:
                missing_features.append(f"{feature_name} (missing joint)")

        elif feature_name == "observation.state.target":
            if "target_cartesian" not in obs:
                missing_features.append(f"{feature_name} (missing target_cartesian)")

        elif feature_name.startswith("observation.state.sensor_"):
            sensor_name = feature_name.split(".")[-1].replace("sensor_", "")
            if sensor_name not in obs:
                missing_features.append(f"{feature_name} (expected key: {sensor_name})")
        else:
            missing_features.append(feature_name)

    if missing_features:
        logger.error(f"Missing required features: {missing_features}")
        return False

    return True


def numpy_obs_to_torch(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy observations to torch tensors for policy inference.

    This function takes a dictionary of observations where each value is a numpy array,
    and converts them to torch tensors. It handles different types of observations
    such as state, task, and images.

    Args:
        obs (Dict[str, Any]): Dictionary of observations with numpy arrays.

    Returns:
        Dict[str, Any]: Dictionary of observations with torch tensors.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for tensor conversion")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_torch = {}

    for key, value in obs.items():
        if key.startswith("observation.state"):
            obs_torch[key] = torch.from_numpy(value).unsqueeze(0).to(device).float()
        elif key.startswith("task"):
            obs_torch[key] = value  # Keep task values as-is
        elif key.startswith("observation.images"):
            obs_torch[key] = (
                torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
            )

    return obs_torch


if __name__ == "__main__":
    # Example usage
    env_config = make_env_config("right_aloha_franka")
    features = get_features(env_config, ctrl_type="cartesian", use_video=True)

    rich.print(features)
