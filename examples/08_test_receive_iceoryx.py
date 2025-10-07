"""Example of receiving observation data via iceoryx."""

# WARNING: This is a protyping script and not meant for usage for now.
import logging
import time

import iceoryx2 as iox2
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

obs_struct_cls = env.create_obs_struct_for_env()
service_name = "example_obs_service"


node = iox2.NodeBuilder.new().create(iox2.ServiceType.Ipc)
service = (
    node.service_builder(iox2.ServiceName.new(service_name))
    .publish_subscribe(obs_struct_cls)
    .open_or_create()
)
subscriber = service.subscriber_builder().create()

time.sleep(1.0 / 100.0)
last_sample_time = time.perf_counter()
while True:
    sample = subscriber.receive()

    if sample is None:
        continue
    elapsed = time.perf_counter() - last_sample_time
    last_sample_time = time.perf_counter()
    data = sample.payload()
    print(f"Received data after {elapsed * 1000:.1f} ms: {data}")
    time.sleep(1.0 / 100.0)
