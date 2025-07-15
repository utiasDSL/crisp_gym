"""Configuration path for Crisp Gym."""

import os
from pathlib import Path

CRISP_CONFIG_PATH = os.environ.get("CRISP_CONFIG_PATH")
if CRISP_CONFIG_PATH is None:
    raise EnvironmentError(
        "Environment variable 'CRISP_CONFIG_PATH' is not set. Please run:\n"
        "  export CRISP_CONFIG_PATH=/path/to/config"
    )
CRISP_CONFIG_PATH = Path(CRISP_CONFIG_PATH)
