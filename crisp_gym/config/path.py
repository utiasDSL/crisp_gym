"""Configuration path for Crisp Gym."""

import os
import warnings
from importlib.resources import files
from pathlib import Path
from typing import List, Optional

CRISP_CONFIG_PATH_STR = os.environ.get("CRISP_CONFIG_PATH")
CRISP_CONFIG_PATHS: List[Path] = []
CRISP_CONFIG_PATH: Path


def _parse_config_paths(path_str: str) -> List[Path]:
    """Parse colon-separated config paths and validate they exist."""
    paths = []
    for path_part in path_str.split(":"):
        path_part = path_part.strip()
        if path_part:
            path = Path(path_part)
            if path.exists():
                paths.append(path)
            else:
                warnings.warn(
                    f"CRISP configuration path '{path}' does not exist and will be ignored."
                )
    return paths


default_path = Path(str(files("crisp_py").joinpath("config")))
CRISP_CONFIG_PATHS = [default_path]
CRISP_CONFIG_PATH = default_path

if CRISP_CONFIG_PATH_STR is not None:
    set_paths = _parse_config_paths(CRISP_CONFIG_PATH_STR)
    if not set_paths:
        raise FileNotFoundError(
            f"No valid CRISP configuration paths found in '{CRISP_CONFIG_PATH_STR}'. "
            "Please ensure at least one path exists and is accessible."
        )

    set_paths.reverse()
    CRISP_CONFIG_PATHS.extend(set_paths)
    CRISP_CONFIG_PATHS.reverse()  # Keep the order as specified in the environment variable
    # For backward compatibility, use the first path
    CRISP_CONFIG_PATH = CRISP_CONFIG_PATHS[0]


def find_config(filename: str) -> Optional[Path]:
    """Find a config file in the CRISP config paths.

    Args:
        filename: Name of the config file to find

    Returns:
        Path to the first matching config file, or None if not found
    """
    for config_path in CRISP_CONFIG_PATHS:
        file_path = config_path / filename
        if file_path.exists():
            return file_path
    return None
