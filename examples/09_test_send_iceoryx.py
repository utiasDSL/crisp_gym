"""Example testing environment with iceoryx for ipc communication."""

# WARNING: This is a protyping script and not meant for usage for now.
import logging
import time

import iceoryx2 as iox2
from gymnasium.spaces import Dict
from rich import print

from crisp_gym.manipulator_env import make_env
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

try:
    env = make_env(namespace="right", env_type="fake_cam_setup", control_type="cartesian")
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

obs_struct_cls = env.create_obs_struct_for_env()
action_struct_cls = env.create_action_struct_for_env()


# %%

service_name = "example_obs_service"


node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
service = (
    node.service_builder(iox2.ServiceName.new(service_name))
    .publish_subscribe(obs_struct_cls)
    .open_or_create()
)
publisher = service.publisher_builder().create()

# %%
env.wait_until_ready()

# %%

start_time = time.time()
while time.time() - start_time < 30.0:
    sample = publisher.loan_uninit()

    obs = env.get_obs()
    sample = sample.write_payload(
        obs_struct_cls(**{key: getattr(obs, key) for key in obs_struct_cls.__annotations__.keys()})
    )
    sample.send()

    time.sleep(1.0 / 100.0)
