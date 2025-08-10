"""Define the control type for the environment."""

from enum import Enum


class ControlType(Enum):
    """Enum for control types for a manipulator environment.

    This is needed to handle different control schemes as well as naming conventions.
    """

    JOINT = "joint"
    """Joint control type."""

    CARTESIAN = "cartesian"
    """Cartesian control type."""

    UNDEFINED = "undefined"
    """Undefined control type."""

    @classmethod
    def from_string(cls, ctrl_type: str) -> "ControlType":
        """Convert a string to a ControlType enum."""
        try:
            return cls[ctrl_type.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid control type: {ctrl_type}. Supported types: {[e.value for e in cls]}"
            )

    def controller_name(self) -> str:
        """Get the controller name for the control type."""
        if self == ControlType.JOINT:
            return "joint_impedance_controller"
        elif self == ControlType.CARTESIAN:
            return "cartesian_impedance_controller"
        else:
            raise ValueError(f"No controller available for controller type: {self.value}")
