"""Example testing teleop."""

import logging

from crisp_gym.teleop.teleop_sensor_stream import TeleopStreamedPose
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

teleop = TeleopStreamedPose()
try:
    teleop.wait_until_ready()
    print(teleop.last_pose)
except Exception as e:
    logger.exception(e)
