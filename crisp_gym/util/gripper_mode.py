"""Gripper modes for the CRISP Gym environment."""

from enum import Enum


class GripperMode(Enum):
    """Gripper control modes."""

    NONE = "none"
    ABSOLUTE_CONTINUOUS = "absolute_continuous"
    RELATIVE_CONTINUOUS = "relative_continuous"
    ABSOLUTE_BINARY = "absolute_binary"
    RELATIVE_BINARY = "relative_binary"


def min_action_for_gripper_mode(mode: GripperMode) -> float:
    """Get the minimum action value for the specified gripper mode.

    Args:
        mode (GripperMode): The gripper mode.

    Returns:
        float: The minimum action value.
    """
    if mode == GripperMode.NONE:
        return 0.0
    elif mode in {GripperMode.ABSOLUTE_CONTINUOUS, GripperMode.RELATIVE_CONTINUOUS}:
        return -1.0
    elif mode in {GripperMode.ABSOLUTE_BINARY, GripperMode.RELATIVE_BINARY}:
        return 0.0
    else:
        raise ValueError(f"Unknown gripper mode: {mode}")


def max_action_for_gripper_mode(mode: GripperMode) -> float:
    """Get the maximum action value for the specified gripper mode.

    Args:
        mode (GripperMode): The gripper mode.

    Returns:
        float: The maximum action value.
    """
    if mode == GripperMode.NONE:
        return 0.0
    elif mode in {GripperMode.ABSOLUTE_CONTINUOUS, GripperMode.RELATIVE_CONTINUOUS}:
        return 1.0
    elif mode in {GripperMode.ABSOLUTE_BINARY, GripperMode.RELATIVE_BINARY}:
        return 1.0
    else:
        raise ValueError(f"Unknown gripper mode: {mode}")
