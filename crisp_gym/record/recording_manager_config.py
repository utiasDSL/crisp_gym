"""Configuration classes for recording managers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from crisp_gym.config.path import find_config, list_configs_in_folder


@dataclass(kw_only=True)
class RecordingManagerConfig:
    """Configuration for recording managers.

    This configuration class contains all parameters needed to initialize
    a recording manager, including dataset configuration, recording settings,
    and system parameters.
    """

    # Dataset configuration
    features: Dict[str, Any]
    repo_id: str
    robot_type: str = "Franka"
    resume: bool = False
    fps: int = 30
    num_episodes: int = 3
    push_to_hub: bool = False

    # System configuration
    use_sound: bool = True
    queue_size: int = 16
    writer_timeout: float = 10.0

    # Bag recording configuration
    use_bag_recording: bool = False
    bag_output_path: str = "/tmp/crisp_bags"
    bag_compression: str = "zstd"

    @classmethod
    def from_yaml(cls, yaml_path: Path | str, **overrides) -> "RecordingManagerConfig":  # noqa: ANN003
        """Create a RecordingManagerConfig from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.
            **overrides: Additional keyword arguments to override config values.

        Returns:
            A RecordingManagerConfig instance.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            yaml.YAMLError: If the YAML file is malformed.
            TypeError: If required fields are missing.
        """
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            config_data = {}

        # Apply overrides
        config_data.update(overrides)

        return cls(**config_data)

    def to_yaml(self, yaml_path: Path | str) -> None:
        """Save the configuration to a YAML file.

        Args:
            yaml_path: Path where to save the YAML configuration file.
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, handling non-serializable types
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, (dict, list, str, int, float, bool)) or field_value is None:
                config_dict[field_name] = field_value
            else:
                # For complex types, convert to string representation
                config_dict[field_name] = str(field_value)

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)


def make_recording_manager_config(
    name: str,
    config_path: Path | str | None = None,
    **overrides,  # noqa: ANN003
) -> RecordingManagerConfig:
    """Factory function to create a recording manager configuration based on the type.

    This function allows for both predefined recording manager types and custom YAML configurations.
    It will first check if the type is in the predefined set, and if not, it will look for a YAML config file.

    Args:
        name: Type of recording manager configuration
        config_path: Optional path to YAML config file
        **overrides: Additional parameters to override defaults/YAML values

    Returns:
        RecordingManagerConfig: Configured recording manager instance
    """
    config_class = STRING_TO_CONFIG.get(name.lower())
    if config_class is None:
        # Try to find YAML config if not in predefined types
        config_path = find_config("recording/" + name.lower() + ".yaml")
        if config_path is None:
            raise ValueError(
                f"Unsupported recording manager type: {name}. The list of supported types are: {list_recording_configs()}"
            )
        config_class = RecordingManagerConfig

    if config_path:
        config_path = Path(config_path) if isinstance(config_path, str) else config_path
        return config_class.from_yaml(config_path, **overrides)

    return config_class(**overrides)


def list_recording_configs() -> list[str]:
    """List all available recording manager configurations."""
    predefined = list(STRING_TO_CONFIG.keys())
    other = list_configs_in_folder("recording")
    yaml_configs = [file.stem for file in other if file.suffix == ".yaml"]
    return predefined + yaml_configs


STRING_TO_CONFIG: Dict[str, type[RecordingManagerConfig]] = {}
