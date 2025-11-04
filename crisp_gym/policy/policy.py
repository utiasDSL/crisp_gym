"""Abstract base class and registry for Policies."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, TypeAlias

import numpy as np

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


def make_policy(name, *args, **kwargs) -> Policy:  # noqa: ANN001, ANN002, ANN003
    """Factory function to create a policy instance by name.

    Args:
        name (str): The name of the policy to create.
        *args: Positional arguments to pass to the policy constructor.
        **kwargs: Keyword arguments to pass to the policy constructor.

    Returns:
        Policy: An instance of the requested policy.
    """
    policy_cls = policy_registry.get(name)
    if policy_cls is None:
        raise ValueError(f"Policy '{name}' is not registered.")
    return policy_cls(*args, **kwargs)
