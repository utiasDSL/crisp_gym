"""This module provides functions to generate and convert features for the LeRobotDataset.

It defines the features used by the dataset, including camera images, proprioceptive state,
sensors, and actions based on the environment configuration and control type. It also provides
utilities for converting observations between different formats.
"""

from typing import Any, Dict

import gymnasium
import numpy as np
import rich

from crisp_gym.manipulator_env import ManipulatorBaseEnv, make_env
from crisp_gym.util.control_type import ControlType

try:
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
except ImportError:
    raise ImportError(
        "The 'lerobot' package is required to use this function. "
        "Please use a lerobot environment 'pixi shell -e <rosdistro>-lerobot'."
    )

import logging

logger = logging.getLogger(__name__)


def get_features(
    env: ManipulatorBaseEnv,
    use_video: bool = True,
    ignore_keys: list[str] = None,
) -> Dict[str, Dict]:
    """Get the features used by LeRobotDataset.

    Args:
        env (ManipulatorBaseEnv): The environment configuration object.
        ctrl_type (str): The type of control used, either "joint" or "cartesian". Defaults to "cartesian".
        use_video (bool): Whether to include video features. Defaults to True.
        ignore_keys (list[str], optional): List of observation keys to ignore. Defaults to None.
    """
    if not CODEBASE_VERSION.startswith("v2"):
        logger.warning(
            "Feature generation for LeRobot has been implemented for version 2.x of LeRobotDataset. Expect unexpected behaviour for other versions."
        )

    ctrl_dims: dict[ControlType, list[str]] = {
        ControlType.JOINT: [f"joint_{idx}" for idx in range(env.config.robot_config.num_joints())]
        + ["gripper"],
        ControlType.CARTESIAN: ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    }

    if env.ctrl_type not in ctrl_dims:
        raise ValueError(
            f"Control type {env.ctrl_type} not supported. Supported types: {list(ctrl_dims.keys())}"
        )

    # Feature configuration
    if not isinstance(env.observation_space, gymnasium.spaces.Dict):
        raise ValueError(
            "The observation space must be a gymnasium Dict space. "
            "This function is designed for environments with structured observations."
        )

    features = {}
    state_feature_length = 0
    state_feature_names = []

    for feature_key in env.observation_space.keys():
        if ignore_keys and feature_key in ignore_keys:
            continue

        if feature_key.startswith("observation.state"):
            # Proprioceptive state features
            feature_shape = env.observation_space[feature_key].shape
            if feature_shape is None:
                raise ValueError(f"Feature shape for {feature_key} should not be None!")

            names = []
            if "joint" in feature_key:
                names = ctrl_dims[ControlType.JOINT][:-1]  # Exclude gripper from joint state
            elif "cartesian" in feature_key:
                names = ctrl_dims[ControlType.CARTESIAN][
                    :-1
                ]  # Exclude gripper from cartesian state
            elif "gripper" in feature_key:
                names = ["gripper"]
            elif "target" in feature_key:
                names = ["target_" + dim for dim in ctrl_dims[env.ctrl_type][:-1]]
            else:
                n: int = feature_shape[0] if feature_shape is not None else 1
                feature_key_name = feature_key.split(".")[-1]
                names = [f"{feature_key_name}_{i}" for i in range(n)]

            features[feature_key] = {
                "dtype": "float32",
                "shape": feature_shape,
                "names": names,
            }
            state_feature_length += int(np.prod(feature_shape))
            state_feature_names += names

        elif feature_key.startswith("task"):
            continue  # Task features are handled separately

        elif feature_key.startswith("observation.images."):
            features[feature_key] = {
                "dtype": "image",
                "shape": env.observation_space[feature_key].shape,
                "names": ["height", "width", "channels"],
            }
            if use_video:
                original_feature_key = feature_key
                feature_key = feature_key.replace("images", "video")
                features[feature_key] = {
                    "dtype": "video",
                    "shape": env.observation_space[original_feature_key].shape,
                    "names": ["height", "width", "channels"],
                    "video_info": {
                        "video.fps": env.config.control_frequency,
                        "video.codec": "av1",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                }

    image_resolutions = [
        feature["shape"]
        for key, feature in features.items()
        if key.startswith("observation.images.")
    ]
    if len(set(image_resolutions)) > 1:
        logger.warning(
            "Images have different resolutions. This might cause issues with the current policies available in LeRobot. "
            "For now, they only support images with the same resolution. "
            "Please ensure all images have the same resolution."
        )
    # Combined state feature
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (int(state_feature_length),),
        "names": state_feature_names,
    }

    # Action
    features["action"] = {
        "dtype": "float32",
        "shape": (len(ctrl_dims[env.ctrl_type]),),
        "names": ctrl_dims[env.ctrl_type],
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
    env = make_env("right_aloha_franka")
    features = get_features(env, use_video=True)

    rich.print(features)
