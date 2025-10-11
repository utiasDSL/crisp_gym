"""Example script to convert MCAP files to LeRobot format."""

import argparse

import numpy as np
from crisp_py.robot import Pose
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mcap.reader import NonSeekingReader
from mcap_ros2.reader import read_ros2_messages
from rich import print
from rich.progress import track
from rich.traceback import install

from crisp_gym.manipulator_env import make_env
from crisp_gym.util.lerobot_features import get_features

install(show_locals=True)

# %%
parser = argparse.ArgumentParser(
    description="Convert MCAP files to LeRobot format and upload to Hugging Face Hub.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--mcap_file", type=str, help="Path to the MCAP file.")
parser.add_argument("--repo_id", type=str, help="Hugging Face repo ID.")

args = parser.parse_args()

# %%
mcap_file = args.mcap_file or "/home/daniel/Downloads/episode_0000_0.mcap"
repo_id = args.repo_id or "danielsanjosepro/test_repo"
fps = 15

# %%
env = make_env("hacked_right_aloha_franka", namespace="right")

ignore_keys = [
    "observation.state.target",
    "observation.state",
]

features = get_features(env, ignore_keys=ignore_keys)
topics_to_features = env.get_topics_to_features()
topics_to_features["/action"] = "action"

filtered_features = {}
for topic, feature in topics_to_features.items():
    if feature in features.keys():
        filtered_features[feature] = features[feature].copy()

# %%

try:
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
    )
except Exception as e:
    env.close()
    raise e

# %%
latest_observation: dict[str, np.ndarray | None] = {
    feature: None for feature in filtered_features.keys()
}

# %%


def get_message_count(file_path: str) -> int:
    """Get the total number of messages in an MCAP file."""
    with open(file_path, "rb") as f:
        reader = NonSeekingReader(f)
        summary = reader.get_summary()
        if summary is None:
            raise ValueError("No summary found in MCAP file")
    return summary.statistics.message_count if summary.statistics else 0


def ros_msg_to_observation(msg) -> np.ndarray:  # noqa: ANN001
    """Convert a ROS message to an observation."""
    if "camera" in msg.channel.topic:
        return env.cameras[0].ros_msg_to_image(msg.ros_msg)
    if "pose" in msg.channel.topic:
        obs = (
            Pose.from_ros_msg(msg.ros_msg)
            .to_array(env.config.orientation_representation)
            .astype(np.float32)
        )
        return obs
    if "joint_states" in msg.channel.topic and "gripper" not in msg.channel.topic:
        return env.robot.ros_msg_to_joint(msg.ros_msg)
    if "action" in msg.channel.topic:
        return np.array(msg.ros_msg.data, dtype=np.float32)
    if "gripper" in msg.channel.topic:
        return np.array([env.gripper.ros_msg_to_gripper_value(msg.ros_msg)], dtype=np.float32)
    raise ValueError(f"Unknown topic: {msg.channel.topic}")


print("Processing MCAP file: " + mcap_file)
message_count = get_message_count(mcap_file)
print("Total messages in MCAP file: " + str(message_count))
print("Converting to LeRobot format and uploading to hub: " + repo_id)

try:
    for msg in track(
        read_ros2_messages(source=mcap_file, topics=[*topics_to_features.keys()]),
        total=message_count,
        description="[Ctrl-C to cancel] Processing MCAP file:",
    ):
        latest_observation[topics_to_features[msg.channel.topic]] = ros_msg_to_observation(msg)
        if "action" in msg.channel.topic:
            if any([observation is None for observation in latest_observation.values()]):
                continue
            dataset.add_frame(latest_observation, task="random")
except KeyboardInterrupt:
    print("\nProcess interrupted by user. Saving progress...")
except Exception as e:
    print(f"\nError processing MCAP file: {e}. Saving progress...")

# %%
try:
    dataset.save_episode()
    dataset.push_to_hub()
except Exception as e:
    env.close()
    raise e

print("Closing environment.")
env.close()
