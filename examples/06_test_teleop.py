"""Example testing teleop."""

import logging

from crisp_gym.teleop.teleop_robot import make_leader
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

try:
    leader = make_leader("minimal_teleop")
except Exception as e:
    logger.exception(e)
