"""Example testing environment."""

import logging

from gymnasium.spaces import Dict
from rich import print

from crisp_gym.manipulator_env import make_env
from crisp_gym.util.lerobot_features import get_features
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

try:
    env = make_env("minimal_env", control_type="cartesian")
except Exception as e:
    logger.exception(e)
    env = None

# %%
if env is None:
    raise ValueError("The environment could not be created. Please check the environment name.")

if not isinstance(env.observation_space, Dict):
    raise ValueError("The environment does not have an observation space.")
print([key for key in env.observation_space.keys()])

# %%

features = get_features(env)
print(features)
