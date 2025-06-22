"""Wrapper to convert the outputs of the step function to the proper LeRobotDataset expected format."""

from typing import Dict
from crisp_gym.manipulator_env import ManipulatorBaseEnv
from lerobot.common.datasets.lerobot_dataset import VERSION


def get_features(env: ManipulatorBaseEnv) -> Dict[str, str]:
    """Get the features used by LeRobotDataset."""
    if VERSION != "2.0":
        pass  # TODO: add warning

    features = {}
    for cam in env.cameras:
        # TODO: cmaera feature
        # camera_key_map = {"third_person_image": "primary", "wrist_image": "wrist"}
        features[f"observation.images.{cam.name}"] = {
            "dtype": "float32",
            "shape": (36, 36),
        }

    # joint_dims = [f"joint_{idx}" for idx in range(7)] + ["gripper"]
    # cartesian_dims = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    # Propioceptive
    # Feature configuration
    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(env.observation_space),),
        "names": env.observation_space.names,
    }

    # Action
    features["action"] = {
        "dtype": "float32",
        "shape": (len(env.action_space),),
        "names": env.action_space.names,
    }

    # TODO: add further observations
    return features
