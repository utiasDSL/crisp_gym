"""Example script to convert MCAP files to LeRobot format."""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mcap_ros2.reader import read_ros2_messages

from crisp_gym.manipulator_env import make_env
from crisp_gym.util.lerobot_features import get_features

# %% Setup parameters
mcap_file = "/home/daniel/crisp_bags/episode_0000/episode_0000_0.mcap"
repo_id = "danielsanjosepro/test_high_frequency"
fps = 100

try:
    from rich import print
except ImportError:
    pass

# %%
env = make_env("right_aloha_franka", namespace="right")

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
    repo_id=repo_id + "v1",
    fps=fps,
    features=features,
    use_videos=True,
)

# %%
latest_observation = {feature: None for feature in filtered_features.keys()}

print("Latest observation:")
print(latest_observation)
print([*topics_to_features.keys()])
# %%
for msg in read_ros2_messages(source=mcap_file, topics=[*topics_to_features.keys()]):
    if "camera" in msg.channel.topic:
        camera_img = env.cameras[0].ros_msg_to_image(msg.ros_msg)
        latest_observation[topics_to_features[msg.channel.topic]] = camera_img
        continue
    latest_observation[topics_to_features[msg.channel.topic]] = msg.ros_msg

# %%
print(latest_observation["observation.images.wrist"])
