"""Wrapper to convert the outputs of the step function to the proper LeRobotDataset expected format."""

from typing import Dict

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION

from crisp_gym.manipulator_env import ManipulatorBaseEnv


def get_features(env: ManipulatorBaseEnv, ctrl_type: str = 'cartesian') -> Dict[str, Dict]:
    """Get the features used by LeRobotDataset."""
    if not CODEBASE_VERSION.startswith("v2"):
        print(
            "WARNING: this function has been implemented for version 2.x of LeRobotDataset. Expect unexpected behaviour for other versions."
        )

    ctrl_dims = {
        'joint': [f"joint_{idx}" for idx in range(7)] + ["gripper"],
        'cartesian': ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    }

    if not ctrl_type in ctrl_dims:
        raise ValueError(f"Control type {ctrl_type} not supported. Supported types: {list(ctrl_dims.keys())}")

    features = {}
    for cam in env.cameras:
        # TODO: cmaera feature
        # camera_key_map = {"third_person_image": "primary", "wrist_image": "wrist"}
        features[f"observation.images.{cam.config.camera_name}"] = {
            "dtype": "image",
            "shape": (*cam.config.resolution, 3),
            "names": ["height", "width", "channels"],
        }

    # Propioceptive
    # Feature configuration
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(ctrl_dims[ctrl_type]),),
        "names": ctrl_dims[ctrl_type]
    }

    # Action
    features["action"] = {
        "dtype": "float32",
        "shape": env.action_space.shape,
        "names": ctrl_dims[ctrl_type],
    }

    # TODO: add further observations

    return features
