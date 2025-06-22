"""Wrapper to convert the outputs of the step function to the proper LeRobotDataset expected format."""

from typing import Dict

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION

from crisp_gym.manipulator_env import ManipulatorBaseEnv


def get_features(env: ManipulatorBaseEnv) -> Dict[str, Dict]:
    """Get the features used by LeRobotDataset."""
    if not CODEBASE_VERSION.startswith("v2"):
        print(
            "WARNING: this function has been implemented for version 2.x of LeRobotDataset. Expect unexpected behaviour for other versions."
        )

    features = {}
    for cam in env.cameras:
        # TODO: cmaera feature
        # camera_key_map = {"third_person_image": "primary", "wrist_image": "wrist"}
        features[f"observation.images.{cam.config.camera_name}"] = {
            "dtype": "float32",
            "shape": cam.config.resolution,
        }

    # joint_dims = [f"joint_{idx}" for idx in range(7)] + ["gripper"]
    cartesian_dims = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    # Propioceptive
    # Feature configuration
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(cartesian_dims),),
        "names": cartesian_dims,  # TODO: deal with other types
    }

    # Action
    features["action"] = {
        "dtype": "float32",
        "shape": env.action_space.shape,
        "names": cartesian_dims,  # TODO: deal with other types
    }

    # TODO: add further observations

    return features
