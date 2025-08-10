"""Configuration path for Crisp Gym."""

import os
from pathlib import Path

CRISP_CONFIG_PATH_STR = os.environ.get("CRISP_CONFIG_PATH")
CRISP_CONFIG_PATH: Path

if CRISP_CONFIG_PATH_STR is None:
    from importlib.resources import files

    CRISP_CONFIG_PATH_STR = str(files("crisp_py").joinpath("config"))
    CRISP_CONFIG_PATH = Path(CRISP_CONFIG_PATH_STR)
else:
    CRISP_CONFIG_PATH: Path = Path(CRISP_CONFIG_PATH_STR)
    if not CRISP_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"CRISP configuration path '{CRISP_CONFIG_PATH}' does not exist. "
            "Please ensure the path is correct and accessible."
        )
