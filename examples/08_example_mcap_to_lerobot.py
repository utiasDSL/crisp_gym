"""Example script to convert MCAP files to LeRobot format."""

import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mcap_ros2.reader import read_ros2_messages

from crisp_gym.manipulator_env import make_env
from crisp_gym.util.lerobot_features import get_features

# %% Setup parameters
mcap_file = "/tmp/crisp_bags/episode_0000/episode_0000_0.mcap"
repo_id = "danielsanjosepro/test_high_frequency"
fps = 100

try:
    from rich import print
except ImportError:
    pass

# %%
env = make_env("simple_cam")

features = get_features(env)
topics_to_features = env.get_topics_to_features()

topics_to_features["/action"] = "action"

filtered_features = {}
for topic, feature in topics_to_features.items():
    if feature in features.keys():
        filtered_features[feature] = features[feature].copy()

print("Filtered features:")
print(filtered_features)
# %%

dataset = LeRobotDataset.create(
    repo_id=repo_id + "v2",
    fps=fps,
    features=features,
    use_videos=True,
)

# %%
latest_observation = {feature: None for feature in filtered_features.keys()}

print("Latest observation:")
print(latest_observation)
# %%
for msg in read_ros2_messages(source=mcap_file, topics=[*topics_to_features.keys()]):
    pass
