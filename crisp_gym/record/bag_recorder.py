"""ROS2 bag recording functionality for crisp_gym recording manager."""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

import rclpy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rclpy.node import Node

from crisp_gym.record.recording_manager_config import RecordingManagerConfig

logger = logging.getLogger(__name__)


class BagRecorder:
    """Handles ROS2 bag recording and conversion to LeRobot datasets."""

    def __init__(self, config: RecordingManagerConfig, node: Node | None = None) -> None:
        """Initialize the bag recorder.

        Args:
            config: Recording manager configuration
            node: Optional ROS2 node for additional functionality
        """
        self.config = config
        self.node = node
        self.current_bag_path: Path | None = None
        self.recording_process: subprocess.Popen | None = None
        self.topics_to_record: list[str] = []
        self.episode_metadata: Dict[str, Any] = {}

    def set_topics_to_record(self, topics: list[str]) -> None:
        """Set the topics to record - should be called by the environment.

        Args:
            topics: List of ROS2 topic names to record
        """
        self.topics_to_record = topics
        logger.info(f"Set topics to record: {topics}")

    def start_episode_recording(self, episode_id: int) -> Path:
        """Start recording a new bag for the current episode.

        Args:
            episode_id: The episode number/ID

        Returns:
            Path to the bag directory

        Raises:
            ValueError: If no topics are specified for recording
            RuntimeError: If ROS2 is not available or bag recording fails
        """
        if not self.topics_to_record:
            raise ValueError(
                "No topics specified for recording. Call set_topics_to_record() first."
            )

        if not rclpy.ok():
            raise RuntimeError("ROS2 is not initialized. Cannot record bags.")

        bag_output_path = getattr(self.config, "bag_output_path", "/tmp/crisp_bags")

        bag_dir = Path(bag_output_path)
        bag_dir.mkdir(parents=True, exist_ok=True)

        bag_dir = bag_dir / f"episode_{episode_id:04d}"

        cmd = (
            [
                "ros2",
                "bag",
                "record",
                "--output",
                str(bag_dir),
                "--storage",
                "mcap",
            ]
            + self.topics_to_record
            + ["tf", "tf_static"]  # Always record TF topics
            + ["action"]  # Always record action topic (if exists)
        )

        logger.info(f"Starting bag recording for episode {episode_id}")
        logger.debug(f"Recording command: {' '.join(cmd)}")

        try:
            self.recording_process = subprocess.Popen(cmd, text=True)
            self.current_bag_path = bag_dir
            self.episode_metadata = {
                "episode_id": episode_id,
                "topics": self.topics_to_record.copy(),
            }

            logger.info(f"Started bag recording to: {bag_dir}")
            return bag_dir

        except subprocess.SubprocessError as e:
            logger.error(f"Failed to start bag recording: {e}")
            raise RuntimeError(f"Failed to start bag recording: {e}") from e

    def stop_episode_recording(self) -> None:
        """Stop the current bag recording."""
        if self.recording_process is None:
            logger.warning("No active bag recording to stop")
            return

        logger.info("Stopping bag recording")

        try:
            self.recording_process.terminate()
            _, stderr = self.recording_process.communicate(timeout=10.0)
            if self.recording_process.returncode != 0:
                logger.warning(
                    f"Bag recording process exited with code {self.recording_process.returncode}"
                )
                if stderr:
                    logger.warning(f"Bag recording stderr: {stderr}")
            else:
                logger.info("Bag recording stopped successfully")

        except subprocess.TimeoutExpired:
            logger.warning("Bag recording did not stop gracefully, forcing termination")
            self.recording_process.kill()
            self.recording_process.wait()
        except Exception as e:
            logger.error(f"Error stopping bag recording: {e}")
        finally:
            self.recording_process = None

    def convert_bag_to_lerobot(
        self,
        bag_path: Path,
        dataset: LeRobotDataset,
        task: str = "No task",
    ) -> None:
        """Convert a bag to LeRobot dataset format.

        Args:
            bag_path: Path to the recorded bag directory
            dataset: LeRobot dataset to add the episode to
            task: Task label for the episode

        Note:
            This is a placeholder implementation. The actual conversion would need to:
            1. Parse the bag file using rosbag2_py
            2. Extract observation and action data from the recorded topics
            3. Synchronize timestamps across topics
            4. Convert to the format expected by LeRobot dataset
        """
        logger.info(f"Converting bag {bag_path} to LeRobot dataset")

        # TODO: Implement actual bag parsing and conversion
        # This would involve:
        # - Using rosbag2_py to read the bag
        # - Mapping ROS topics to LeRobot observation/action fields
        # - Handling timestamp synchronization
        # - Converting data formats (e.g., ROS Image to numpy arrays)

        logger.warning("Bag to LeRobot conversion not yet implemented - this is a placeholder")

        # For now, just log the metadata
        logger.info(f"Episode metadata: {self.episode_metadata}")
        logger.info(f"Task: {task}")
        logger.info(f"Bag path: {bag_path}")

    def cleanup(self) -> None:
        """Clean up resources and stop any active recordings."""
        if self.recording_process is not None:
            logger.info("Cleaning up active bag recording")
            self.stop_episode_recording()

    def delete_bag(self, bag_path: Path) -> None:
        """Delete a recorded bag directory.

        Args:
            bag_path: Path to the bag directory to delete
        """
        if bag_path.exists() and bag_path.is_dir():
            for item in bag_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    self.delete_bag(item)
            bag_path.rmdir()
            logger.info(f"Deleted bag directory: {bag_path}")
        else:
            logger.warning(f"Bag path does not exist or is not a directory: {bag_path}")

    def delete_last_bag(self) -> None:
        """Delete the last recorded bag, if it exists."""
        if self.current_bag_path:
            self.delete_bag(self.current_bag_path)
            self.current_bag_path = None
        else:
            logger.warning("No current bag path to delete")

    def is_recording(self) -> bool:
        """Check if bag recording is currently active.

        Returns:
            True if recording is active, False otherwise
        """
        return self.recording_process is not None and self.recording_process.poll() is None
