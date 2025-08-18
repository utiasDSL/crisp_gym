"""Class defining the teleoperation leader robot in a leader-follower setup."""

from crisp_py.gripper import Gripper
from crisp_py.robot import Robot

from crisp_gym.teleop.teleop_robot_config import TeleopRobotConfig, make_leader_config


class TeleopRobot:
    """Class defining the teleoperation leader robot in a leader-follower setup.

    This class encapsulates the functionality for controlling a leader robot in a
    teleoperation scenario, allowing for interaction with a follower robot or environment.
    """

    def __init__(self, config: TeleopRobotConfig, namespace: str = ""):
        """Initialize the TeleopRobot with a leader robot and its gripper.

        Args:
            config (TeleopRobotConfig): Configuration for the teleoperation leader robot,
                including the leader robot and its gripper configurations.
            namespace (str, optional): Namespace for the leader robot. Defaults to an empty string.
        """
        self.config = config
        self.robot = Robot(
            robot_config=config.leader, namespace=config.leader_namespace or namespace
        )
        config.leader_gripper.index = 0
        if config.leader_gripper is not None:
            self.gripper = Gripper(
                gripper_config=config.leader_gripper,
                namespace=config.leader_gripper_namespace or f"{namespace}/gripper",
            )
        else:
            self.gripper = None

    def wait_until_ready(self):
        """Wait until the leader robot and its gripper are ready."""
        self.robot.wait_until_ready()
        if self.gripper is not None:
            self.gripper.wait_until_ready()

    def prepare_for_teleop(self, home: bool = True, blocking: bool = True):
        """Prepare the leader robot for teleoperation.

        This method sets the leader robot to a ready state for teleoperation,
        ensuring that it is in a suitable configuration to receive commands.
        """
        if home:
            self.robot.home(blocking=blocking)

        self.robot.cartesian_controller_parameters_client.load_param_config(
            file_path=self.config.gravity_compensation_controller
        )
        self.robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")

        if self.gripper is not None and not self.config.disable_gripper_torque:
            self.gripper.disable_torque()


def make_leader(name: str, namespace: str = "") -> TeleopRobot:
    """Create a TeleopRobot instance using the specified configuration.

    Args:
        name (str): The name of the robot configuration to use.
        namespace (str, optional): Namespace for the leader robot. Defaults to "".

    Returns:
        TeleopRobot: A fully initialized TeleopRobot instance.

    Raises:
        ValueError: If the specified robot configuration name is not supported.
    """
    config = make_leader_config(name)
    return TeleopRobot(config=config, namespace=namespace)
