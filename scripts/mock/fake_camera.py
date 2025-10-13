"""Fake camera publisher node for ROS2 that generates and publishes noise images and camera info."""
#!/usr/bin/env python3  # noqa: D100

import argparse
import threading

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Header


class FakeCameraPublisher(Node):  # noqa: D101
    def __init__(  # noqa: D107
        self,
        camera_name: str = "camera",
        image_width: int = 640,
        image_height: int = 640,
        fps: float = 30.0,
        namespace: str | None = None,
    ):
        super().__init__(f"{camera_name}_publisher", namespace=namespace)

        self.camera_name = camera_name
        self.image_width = image_width
        self.image_height = image_height
        self.fps = fps

        # Publishers
        self.image_pub = self.create_publisher(
            CompressedImage,
            f"{camera_name}/image_raw/compressed",
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo,
            f"{camera_name}/camera_info",
            qos_profile_sensor_data,
            callback_group=ReentrantCallbackGroup(),
        )

        # Shared buffer for compressed image data
        self._compressed_image_msg = None
        self._image_lock = threading.Lock()

        # Create callback groups for parallel execution
        self.generation_group = ReentrantCallbackGroup()
        self.publishing_group = ReentrantCallbackGroup()

        # Timer for generating and compressing images
        self.generation_timer = self.create_timer(
            1.0 / self.fps,
            self.generate_and_compress_image,
            callback_group=self.generation_group,
        )

        # Timer for publishing (same frequency)
        self.publish_timer = self.create_timer(
            1.0 / self.fps,
            self.publish_data,
            callback_group=self.publishing_group,
        )

        # Camera info message (placeholder values)
        self.camera_info_msg = self._create_camera_info()

        self.get_logger().info(
            f"Fake camera '{camera_name}' and namespace '{namespace}' publisher started at {self.fps} Hz"
        )
        self.get_logger().info("Publishing on topics:")
        self.get_logger().info(f"  - {self.image_pub.topic_name}")
        self.get_logger().info(f"  - {self.camera_info_pub.topic_name}")

    def _create_camera_info(self) -> CameraInfo:
        camera_info = CameraInfo()
        camera_info.width = self.image_width
        camera_info.height = self.image_height
        camera_info.distortion_model = "plumb_bob"

        # Placeholder camera matrix (focal length = 500, principal point at center)
        camera_info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 320.0, 0.0, 0.0, 1.0]

        # Placeholder distortion coefficients (no distortion)
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Rectification matrix (identity for no rectification)
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Projection matrix
        camera_info.p = [500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 320.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        return camera_info

    def _generate_noise_image(self) -> np.ndarray:
        # Generate random noise image (RGB)
        noise = np.random.randint(0, 256, (self.image_height, self.image_width, 3), dtype=np.uint8)
        return noise

    def generate_and_compress_image(self):
        """Generate noise image and compress it (runs in separate thread)."""
        timestamp = self.get_clock().now().to_msg()

        # Generate noise image
        noise_image = self._generate_noise_image()

        # Compress image to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        success, encoded_image = cv2.imencode(".jpg", noise_image, encode_param)

        if success:
            # Create header
            header = Header()
            header.stamp = timestamp
            header.frame_id = f"{self.camera_name}_optical_frame"

            # Create compressed message
            compressed_msg = CompressedImage()
            compressed_msg.header = header
            compressed_msg.format = "jpeg"
            compressed_msg.data = encoded_image.tobytes()

            # Thread-safe update of shared buffer
            with self._image_lock:
                self._compressed_image_msg = compressed_msg

    def publish_data(self):
        """Publish the latest compressed image and camera info (runs in separate thread)."""
        timestamp = self.get_clock().now().to_msg()

        # Create header for camera info
        header = Header()
        header.stamp = timestamp
        header.frame_id = f"{self.camera_name}_optical_frame"

        # Publish compressed image if available
        with self._image_lock:
            if self._compressed_image_msg is not None:
                # Update timestamp to current time for publishing
                self._compressed_image_msg.header.stamp = timestamp
                self.image_pub.publish(self._compressed_image_msg)

        # Publish camera info
        self.camera_info_msg.header = header
        self.camera_info_pub.publish(self.camera_info_msg)


def main():
    """Main function to run the fake camera publisher node."""
    parser = argparse.ArgumentParser(description="Fake camera publisher for ROS2")
    parser.add_argument(
        "--name", type=str, default="camera", help="Name of the camera (default: camera)"
    )
    parser.add_argument("--width", type=int, default=640, help="Image width (default: 640)")
    parser.add_argument("--height", type=int, default=640, help="Image height (default: 640)")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30.0)")
    parser.add_argument("--namespace", type=str, default=None, help="ROS namespace (default: None)")

    args = parser.parse_args()

    rclpy.init()

    try:
        fake_camera = FakeCameraPublisher(
            camera_name=args.name,
            image_width=args.width,
            image_height=args.height,
            fps=args.fps,
            namespace=args.namespace,
        )

        # Use MultiThreadedExecutor with 2 threads
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(fake_camera)

        try:
            executor.spin()
        finally:
            executor.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
