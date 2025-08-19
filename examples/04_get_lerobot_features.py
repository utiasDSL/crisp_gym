"""Example showing how to get the features for a LeRobotDataset using a manipulator environment config."""

from crisp_py.sensors.sensor_config import ForceTorqueSensorConfig
from rich import print

from crisp_gym.manipulator_env_config import make_env_config
from crisp_gym.util.lerobot_features import get_features

env_type = "left_aloha_franka"  # Example environment type
env_config = make_env_config(env_type)

force_torque_sensor = ForceTorqueSensorConfig()
env_config.sensor_configs = [force_torque_sensor]

features = get_features(env_config, ctrl_type="cartesian")

print("Features for LeRobotDataset:")
for feature_name, feature_info in features.items():
    print(f"{feature_name}: {feature_info}")
