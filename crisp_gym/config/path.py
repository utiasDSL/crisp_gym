"""Configuration path for Crisp Gym.

This module now imports configuration management from crisp_py for centralized
config management across all CRISP packages.
"""

from pathlib import Path

from crisp_py.config.path import (
    CRISP_CONFIG_PATH,
    CRISP_CONFIG_PATHS,
    find_config,
    list_configs_in_folder,
)

CRISP_GYM_CONFIG_PATH = Path(__file__).parent
CRISP_CONFIG_PATHS.append(CRISP_GYM_CONFIG_PATH)

__all__ = [
    "CRISP_CONFIG_PATH",
    "CRISP_CONFIG_PATHS",
    "find_config",
    "list_configs_in_folder",
]
