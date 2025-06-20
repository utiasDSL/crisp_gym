"""Script showcasing how to record data in Lerobot Format."""

import shutil
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame

cameras = {
    "third_person_image": (256, 256, 3),
    "wrist_image": (256, 256, 3),
}
camera_key_map = {"third_person_image": "primary", "wrist_image": "wrist"}

joint_dims = [f"joint_{idx}" for idx in range(7)] + ["gripper"]
cartesian_dims = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


features = {}
repo_id = "franka_single"
single_task = "pick the lego block."
robot_type = "franka"
fps = 30

features["observation.state"] = {
    "dtype": "float32",
    "shape": (len(cartesian_dims),),
    "names": cartesian_dims,
}

features["action"] = {
    "dtype": "float32",
    "shape": (len(cartesian_dims),),
    "names": cartesian_dims,
}

# for cam, res in cameras.items():
#     features[f"observation.images.{camera_key_map[cam]}"] = {
#         "dtype": 'image',
#         "shape": res,
#         "names": ["height", "width", "channels"],
#     }

if Path(HF_LEROBOT_HOME / repo_id).exists():
    shutil.rmtree(HF_LEROBOT_HOME / repo_id)


dataset = LeRobotDataset.create(
    repo_id=repo_id,
    fps=fps,
    robot_type=robot_type,
    features=features,
    use_videos=False,
)

obs_dict = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "gripper": 0.0,
}

action_dict = {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "gripper": 0.0,
}

action_frame = build_dataset_frame(features, action_dict, prefix="action")
obs_frame = build_dataset_frame(features, obs_dict, prefix="observation.state")
frame = {**obs_frame, **action_frame}

dataset.add_frame(frame, task=single_task)

dataset.save_episode()
