"""Example showing how to get the features for a LeRobotDataset using a manipulator environment config."""

from crisp_py.sensors.sensor_config import SensorConfig
from rich import print
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.manipulator_env_config import make_env_config

env_type = "left_aloha_franka"  # Example environment type
env_config = make_env_config(env_type)

env_config.sensor_configs = [
    SensorConfig(
        name="force_torque",
        shape=(6,),  # Example shape for force/torque sensor
        data_topic="/franka/joint_states",  # Example topic, adjust as needed
    ),
]  # No sensors for this example

features = get_features(env_config, ctrl_type="cartesian")
print("Features for LeRobotDataset:")
for feature_name, feature_info in features.items():
    print(f"{feature_name}: {feature_info}")
