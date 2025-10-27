"""Wrapper to convert the outputs of the step function to the proper LeRobotDataset expected format."""

from typing import Any, Dict
import numpy as np
import torch 

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

def numpy_obs_to_torch(obs: Dict[str, Any], env) -> Dict[str, Any]:
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

    state = np.concatenate([obs["cartesian"][:6], obs["gripper"]])
    batch = {
        "observation.state": torch.from_numpy(state)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32),
        "task": "", # TODO: Add task description if needed
    }
    for cam in env.cameras:
        img = obs[f"{cam.config.camera_name}_image"]
        batch[f"observation.images.{cam.config.camera_name}"] = (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)/ 255
        )

    return batch

