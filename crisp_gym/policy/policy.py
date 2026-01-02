"""Abstract base class and registry for Policies."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, TypeAlias

import numpy as np
import yaml

from crisp_gym.config.path import find_config, list_configs_in_folder

Action: TypeAlias = np.ndarray
Observation: TypeAlias = dict[str, Any]


policy_registry = {}


class Policy(ABC):
    """Abstract base class for a Policy."""

    @abstractmethod
    def make_data_fn(self, *args, **kwargs) -> Callable[[], Tuple[Observation, Action]]:  # noqa: ANN002, ANN003
        """Generate observation and action by communicating with the inference worker."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the policy state."""
        pass

    @abstractmethod
    def shutdown(self):
        """Shutdown the policy and release resources."""
        pass


def register_policy(name: str) -> Callable:
    """Decorator to register a Policy class with a given name."""

    def decorator(cls: Policy) -> Policy:
        policy_registry[name] = cls
        return cls

    return decorator


def make_policy(name_or_config_name, *args, **kwargs) -> Policy:  # noqa: ANN001, ANN002, ANN003
    """Factory function to create a policy instance by name.

    Args:
        name_or_config_name (str): The name of the policy or the config file name.
        *args: Positional arguments to pass to the policy constructor.
        **kwargs: Keyword arguments to pass to the policy constructor, potentially overriding config values.

    Returns:
        Policy: An instance of the requested policy.
    """
    file_path = find_config(
        "policy"
        + "/"
        + name_or_config_name
        + ("" if name_or_config_name.endswith(".yaml") else ".yaml")
    )
    config = {}
    if file_path is not None:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
        name = config.pop("name")
    else:
        name = name_or_config_name

    policy_cls = policy_registry.get(name)
    if policy_cls is None:
        raise ValueError(
            f"Policy '{name}' is not registered. Available policies: {list(policy_registry.keys())}",
            f"{'Make sure the policy is registered and the config file exists: ' + str(file_path) if file_path is not None else ''}",
        )

    return policy_cls(*args, **config, **kwargs)


def list_policy_configs() -> list[str]:
    """List all registered policy names.

    Returns:
        list[str]: A list of registered policy names.
    """
    other = list_configs_in_folder("policy")
    yaml_configs = [file.stem for file in other if file.suffix == ".yaml"]
    if not yaml_configs:
        raise ValueError(
            "No policy configurations found inside 'policy' folder. "
            "Run 'crisp-check-config' to verify the available configs.",
        )
    return yaml_configs
