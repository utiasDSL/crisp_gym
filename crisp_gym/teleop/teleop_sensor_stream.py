"""Class defining the teleoperation for a pose streamer."""

import threading
import time

import rclpy
import rclpy.executors
from crisp_py.utils.geometry import Pose
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32


class TeleopStreamedPose:
    """Class to handle teleoperation using streamed pose and gripper data potentially from a phone, VR device, etc."""

    def __init__(self, namespace: str = ""):
        """Initialize the TeleopStreamedPose class."""
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("pose_streamer", namespace=namespace)

        self._prefix = f"{namespace}_" if namespace else ""

        self._last_pose: Pose | None = None
        self._last_gripper: float | None = None

        # Set this with config
        self._griper_topic = f"/{self._prefix}phone_gripper"
        self._pose_topic = f"/{self._prefix}phone_pose"

        self.node.create_subscription(
            PoseStamped,
            self._pose_topic,
            callback=self._callback_pose,
            callback_group=ReentrantCallbackGroup(),
            qos_profile=qos_profile_sensor_data,
        )

        self.node.create_subscription(
            Float32,
            self._griper_topic,
            callback=self._callback_gripper,
            callback_group=ReentrantCallbackGroup(),
            qos_profile=qos_profile_sensor_data,
        )

        threading.Thread(target=self._spin_node, daemon=True).start()

    @property
    def last_pose(self) -> Pose:
        """Get the last received pose.

        Returns:
            Pose | None: The last received pose or None if no pose has been received yet.
        """
        if self._last_pose is None:
            raise RuntimeError(
                "No pose received yet. Is the teleop device running? Check with 'ros2 topic echo {"
            )
        return self._last_pose

    @property
    def last_gripper(self) -> float:
        """Get the last received gripper value.

        Returns:
            float | None: The last received gripper value or None if no gripper value has been received yet.
        """
        if self._last_gripper is None:
            raise RuntimeError("No gripper value received yet. Is the teleop device running?")
        return self._last_gripper

    def _spin_node(self):
        if not rclpy.ok():
            rclpy.init()
        executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
        executor.add_node(self.node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    def _callback_gripper(self, msg: Float32):
        self._last_gripper = msg.data

    def _callback_pose(self, msg: PoseStamped):
        self._last_pose = Pose.from_ros_msg(msg)

    def is_ready(self) -> bool:
        """Check if the leader robot and its gripper are ready.

        Returns:
            bool: True if both the leader robot and its gripper are ready, False otherwise.
        """
        return self._last_pose is not None

    def wait_until_ready(self, timeout: float = 5.0):
        """Wait until the leader robot and its gripper are ready."""
        start_time = time.time()
        while not self.is_ready() and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                raise TimeoutError("Timed out waiting for the teleop streamer to be ready.")
        if not rclpy.ok():
            raise RuntimeError("ROS2 has been shutdown.")
