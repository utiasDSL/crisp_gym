"""Module for combining hierarchical actions based on different frequencies."""

from dataclasses import dataclass

import numpy as np


@dataclass
class HierarchiechalActionTypeConfig:
    """Configuration for hierarchical action types.

    Args:
        dataset_fps (int): The frames per second (FPS) of the dataset.
        actual_fps (int): The actual FPS of the recorded data.
        relative (bool, optional): Whether the actions are relative or absolute. Defaults to True.
    """

    low_fps: int
    fast_fps: int

    create_low_fps: bool = True

    combine_method: str = "mean"

    def __post_init__(self):
        """Post-initialization checks."""
        if self.low_fps <= 0:
            raise ValueError("Low FPS must be a positive integer.")
        if self.fast_fps <= 0:
            raise ValueError("Fast FPS must be a positive integer.")

        if self.fps_ratio < 1.0:
            raise ValueError(
                f"The low FPS ({self.low_fps}) cannot be higher than the fast FPS ({self.fast_fps})."
            )

        if self.combine_method not in ["mean"]:
            raise ValueError(
                f"Invalid combine method '{self.combine_method}'. Supported methods: 'mean'."
            )

    @property
    def fps_ratio(self) -> float:
        """Ratio of actual FPS to dataset FPS."""
        return self.fast_fps / self.low_fps

    def is_action_processing_required(self) -> bool:
        """Check if action processing is required based on the FPS values."""
        return self.fast_fps > self.low_fps


class HierarchiecalActionCombiner:
    """Combines hierarchical actions based on the provided configuration."""

    def __init__(self, config: HierarchiechalActionTypeConfig):
        """Initialize the HierarchiecalActionCombiner.

        Args:
            config (HierarchiechalActionTypeConfig): Configuration for hierarchical action types.
        """
        self.config = config

        self.action_buffer: list[np.ndarray] = []
        self.observation_buffer: list[np.ndarray] = []

        self._action_dim: int | None = None

    def create_combined_action_if_required(
        self,
        latest_observation: dict[str, np.ndarray],
        latest_action: np.ndarray,
    ) -> np.ndarray | None:
        """Create a combined action based on the latest observation and configuration.

        Note: if the action frequency is the same as the dataset frequency, the action is returned as is.

        Args:
            latest_observation (dict[str, np.ndarray]): The latest observation dictionary.
            latest_action (np.ndarray): The latest action array.

        Returns:
            np.ndarray | None: The combined action array, or None if not enough actions have been collected.
        """
        self._set_action_dim_if_not_set(latest_action)

        if self.config.is_action_processing_required():
            self._fill_buffers(latest_observation, latest_action)

            if not self._low_frequency_action_should_be_generated():
                return None

            combined_action = self._combine_high_frequency_actions_in_one()
            self._empty_buffers()

            return combined_action

        return latest_action

    #   === Private methods ===
    def _set_action_dim_if_not_set(self, latest_action: np.ndarray):
        """Set the action dimension if it is not already set."""
        if self._action_dim is None:
            self._action_dim = latest_action.shape[0]

        assert latest_action.shape[0] == self._action_dim, (
            f"Expected action dimension {self._action_dim}, but got {latest_action.shape[0]}."
        )

    def _empty_buffers(self):
        """Empty the action buffer."""
        self.action_buffer = []
        self.observation_buffer = []

    def _fill_buffers(self, latest_observation: dict[str, np.ndarray], latest_action: np.ndarray):
        """Fill the action and observation buffers with the latest data."""
        self.action_buffer.append(latest_action)
        self.observation_buffer.append(latest_observation["observation.image"])

    def _low_frequency_action_should_be_generated(self) -> bool:
        """Determine if a low-frequency action should be generated based on the FPS ratio."""
        return len(self.action_buffer) >= self.config.fps_ratio

    def _combine_high_frequency_actions_in_one(self) -> np.ndarray:
        """Combine multiple action arrays into a single action array."""
        assert self._action_dim is not None, "Action dimension is not set, cannot combine actions."

        action_pose_or_joint = np.zeros((self._action_dim - 1,), dtype=np.float32)
        action_gripper = 0.0

        for action in self.action_buffer:
            action_pose_or_joint += action[:-1]
            action_gripper = action[-1]  # Use the last gripper value
        return np.concatenate([action_pose_or_joint, [action_gripper]], axis=0)
