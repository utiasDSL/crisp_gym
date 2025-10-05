"""Configuration classes for recording managers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


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
    
    @classmethod
    def from_yaml(cls, yaml_path: Path | str, **overrides) -> "RecordingManagerConfig":
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
            
        with open(yaml_path, 'r') as f:
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
                
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)