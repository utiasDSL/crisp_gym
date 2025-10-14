"""Keyboard event listener for controlling episode recording."""

import logging
import multiprocessing as mp
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import rclpy

# TODO: make this optional, we do not want to depend on lerobot
from lerobot.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pynput import keyboard
from rclpy.executors import SingleThreadedExecutor
from rich import print
from rich.panel import Panel
from std_msgs.msg import Float32MultiArray, String
from typing_extensions import override

from crisp_gym.config.path import find_config
from crisp_gym.record.bag_recorder import BagRecorder
from crisp_gym.record.recording_manager_config import RecordingManagerConfig
from crisp_gym.util import setup_logger
from crisp_gym.util.lerobot_features import concatenate_state_features

logger = logging.getLogger(__name__)


class RecordingManager(ABC):
    """Base class for event listener to control episode recording."""

    def __init__(
        self,
        config: RecordingManagerConfig | None = None,
        topics_to_record: list[str] | None = None,
        **kwargs,  # noqa: ANN003
    ) -> None:
        """Initialize the recording manager.

        Args:
            config: RecordingManagerConfig instance. If provided, other parameters are ignored.
            topics_to_record: List of ROS2 topic names to record if bag recording is enabled.
            **kwargs: Individual parameters for backwards compatibility.
        """
        if not rclpy.ok():
            raise RuntimeError(
                "ROS2 is not initialized. Please initialize ROS2 before using the RecordingManager."
            )
        self.node = rclpy.create_node("recording_manager")
        self._action_publisher = self.node.create_publisher(Float32MultiArray, "action", 10)

        self.config = (
            config
            if config is not None
            else RecordingManagerConfig.from_yaml(
                find_config("recording/default_recording.yaml"), **kwargs
            )
        )

        self.state: Literal[
            "is_waiting",
            "recording",
            "paused",
            "to_be_saved",
            "to_be_deleted",
            "exit",
        ] = "is_waiting"

        self.episode_count = 0

        self.queue = mp.JoinableQueue(self.config.queue_size)
        self.episode_count_queue = mp.Queue(1)
        self.dataset_ready = mp.Event()

        # Initialize bag recorder if enabled
        self.bag_recorder: BagRecorder | None = None
        if self.config.use_bag_recording:
            self.bag_recorder = BagRecorder(self.config)
            self.bag_recorder.set_topics_to_record(
                topics_to_record if topics_to_record is not None else []
            )

        # Start the writer process
        self.writer = mp.Process(
            target=self._writer_proc,
            args=(),
            name="dataset_writer",
            daemon=True,
        )
        self.writer.start()

        threading.Thread(target=self._spin_node, daemon=True).start()

        logger.debug("Instantiated recording manager with config:")
        logger.debug(self.config)

    def _spin_node(self):
        """Spin the ROS2 node in a separate thread."""
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

    @property
    def num_episodes(self) -> int:
        """Return the number of episodes to record."""
        return self.config.num_episodes

    def wait_until_ready(self, timeout: float | None = None) -> None:
        """Wait until the dataset writer is ready."""
        if timeout is None:
            timeout = self.config.writer_timeout

        original_timeout = timeout
        while not self.dataset_ready.is_set():
            logger.debug("Waiting for dataset to be ready...")
            time.sleep(1.0)
            timeout -= 1.0
            if timeout <= 0.0:
                raise TimeoutError(
                    f"Timeout waiting for dataset to be ready after {original_timeout} seconds."
                )

        self.update_episode_count()

    def update_episode_count(self) -> None:
        """Update the episode count from the queue.

        This is useful when resuming from an existing dataset.
        If the queue is empty, it will not change the episode count.
        """
        if not self.episode_count_queue.empty():
            self.episode_count = self.episode_count_queue.get()

    def done(self) -> bool:
        """Return true if we are done recording."""
        return self.state == "exit"

    def register_bag_topics(self, topics: list[str]) -> None:
        """Allow environment to register topics for bag recording.

        Args:
            topics: List of ROS2 topic names to record
        """
        if self.bag_recorder:
            self.bag_recorder.set_topics_to_record(topics)
            logger.info(f"Registered bag topics: {topics}")
        else:
            logger.warning("Bag recording is disabled, ignoring topic registration")

    @abstractmethod
    def get_instructions(self) -> str:
        """Return the instructions to use the recording manager."""
        raise NotImplementedError()

    def _create_dataset(self) -> LeRobotDataset:
        """Factory function to create a dataset object."""
        logger.debug("Creating dataset object.")
        if self.config.resume:
            logger.info(f"Resuming recording from existing dataset: {self.config.repo_id}")
            dataset = LeRobotDataset(
                repo_id=self.config.repo_id, root=HF_LEROBOT_HOME / self.config.repo_id
            )
            if self.config.num_episodes <= dataset.num_episodes:
                logger.error(
                    f"The dataset already has {dataset.num_episodes} recorded. Please select a larger number."
                )
                exit()
            logger.info(
                f"Resuming from episode {dataset.num_episodes} with {self.config.num_episodes} episodes to record."
            )
            self.episode_count_queue.put(dataset.num_episodes - 1)
        else:
            logger.info(
                f"[green]Creating new dataset: {self.config.repo_id}", extra={"markup": True}
            )
            # Clean up existing dataset if it exists
            if Path(HF_LEROBOT_HOME / self.config.repo_id).exists():
                logger.error(
                    f"The repo_id already exists. If you intended to resume the collection of data, then execute this script with the --resume flag. Otherwise remove it:\n'rm -r {str(Path(HF_LEROBOT_HOME / self.config.repo_id))}'."
                )
                raise FileExistsError(
                    f"The repo_id already exists. If you intended to resume the collection of data, then execute this script with the --resume flag. Otherwise remove it:\n'rm -r {str(Path(HF_LEROBOT_HOME / self.config.repo_id))}'."
                )
            dataset = LeRobotDataset.create(
                repo_id=self.config.repo_id,
                fps=self.config.fps,
                robot_type=self.config.robot_type,
                features=self.config.features,
                use_videos=True,
            )
            logger.debug(f"Dataset created with meta: {dataset.meta}")
        return dataset

    def _writer_proc(self):
        """Process to write data to the dataset."""
        logger = logging.getLogger("dataset_writer")
        setup_logger.setup_logging(logging.DEBUG)
        logger.info("Starting dataset writer process.")
        dataset = self._create_dataset()
        self.dataset_ready.set()
        logger.debug(f"Dataset features: {list(self.config.features.keys())}")

        process_episode_count = self.episode_count

        while True:
            msg = self.queue.get()
            logger.debug(f"Received message: {msg['type']}")
            try:
                mtype = msg["type"]

                if mtype == "FRAME":
                    self._add_frame(msg, dataset)

                elif mtype == "START_EPISODE":
                    self._play_sound_if_enabled("start")
                    logger.info(f"Starting episode {process_episode_count}")
                    if self.bag_recorder:
                        self.bag_recorder.start_episode_recording(process_episode_count)

                elif mtype == "STOP_EPISODE":
                    logger.info(f"Stopping episode {process_episode_count}")
                    if self.bag_recorder:
                        logger.info("Stopping bag recording...")
                        self.bag_recorder.stop_episode_recording()

                elif mtype == "SAVE_EPISODE":
                    self._play_sound_if_enabled("save")

                    if self.bag_recorder:
                        logger.info("Converting bag to LeRobot format...")
                        self.bag_recorder.convert_bag_to_lerobot(
                            Path(self.config.bag_output_path), dataset
                        )
                    else:
                        dataset.save_episode()
                    logger.info(
                        f"Episode {process_episode_count} saved to dataset.",
                    )
                    process_episode_count += 1

                elif mtype == "DELETE_EPISODE":
                    self._play_sound_if_enabled("delete")

                    if self.bag_recorder:
                        self.bag_recorder.delete_last_bag()
                    else:
                        dataset.clear_episode_buffer()

                elif mtype == "PUSH_TO_HUB":
                    logger.info("Pushing dataset to Hugging Face Hub...")
                    try:
                        dataset.push_to_hub(repo_id=self.config.repo_id, private=True)
                        logger.info("Dataset pushed to Hugging Face Hub successfully.")
                    except Exception as e:
                        logger.error(
                            f"Failed to push dataset to Hugging Face Hub: {e}",
                            exc_info=True,
                        )
                elif mtype == "SHUTDOWN":
                    logger.info("Shutting down writer process.")
                    break
            except Exception as e:
                logger.exception("Error occured: ", e)
            finally:
                pass

        self.queue.task_done()
        logger.info("Writter process finished.")

    def _play_sound_if_enabled(self, sound_type: Literal["start", "delete", "save"]) -> None:
        if not self.config.use_sound:
            return

        sound_file = ""
        if sound_type == "delete":
            sound_file = "/usr/share/sounds/freedesktop/stereo/suspend-error.oga "
        elif sound_type == "save":
            sound_file = "/usr/share/sounds/freedesktop/stereo/complete.oga "
        elif sound_type == "start":
            sound_file = "/usr/share/sounds/freedesktop/stereo/service-login.oga "

        try:
            subprocess.Popen(
                [
                    "paplay",
                    sound_file,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error(
                f"Failed to play sound for episode completion: {e}",
            )

    def _add_frame(self, msg: dict, dataset: LeRobotDataset) -> None:
        obs, action, task = msg["data"]
        logger.debug(f"Received frame with action: {action} and obs: {obs.keys()}")
        frame = {"action": action.astype(np.float32)}

        for feature_name in self.config.features:
            if feature_name == "action":
                continue
            if feature_name in obs:
                value = obs[feature_name]
                if isinstance(value, np.ndarray) and feature_name.startswith("observation.state"):
                    frame[feature_name] = value.astype(np.float32)
                else:
                    frame[feature_name] = value

        frame["observation.state"] = concatenate_state_features(obs, self.config.features)

        logger.debug(f"Constructed frame with keys: {frame.keys()}")
        dataset.add_frame(frame, task=task)

    def record_episode(
        self,
        data_fn: Callable[[], tuple[dict, dict]],
        task: str,
        on_start: Callable[[], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> None:
        """Record a single episode from user-provided data function.

        Args:
            data_fn: A function that returns (obs, action) at each step.
            task: The task label for the episode.
            on_start: Optional hook called at the start of the episode.
            on_end: Optional hook called at the end (before save/delete).
        """
        try:
            self._wait_for_start_signal()
        except StopIteration:
            logger.info("Recording manager is shutting down.")
            return

        if on_start:
            logger.info("Resetting Environment.")
            on_start()

        logger.info("Started recording episode.")
        self.queue.put({"type": "START_EPISODE"})

        while self.state == "recording":
            frame_start = time.time()

            obs, action = data_fn()

            if obs is None or action is None:
                logger.debug("Data function returned None, skipping frame.")
                # If the data function returns None, skip this frame
                sleep_time = 1 / self.config.fps - (time.time() - frame_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            if not self.config.use_bag_recording:
                self.queue.put({"type": "FRAME", "data": (obs, action, task)})
            else:
                msg = Float32MultiArray()
                msg.data = action.astype(np.float32).tolist()
                self._action_publisher.publish(msg)

            sleep_time = 1 / self.config.fps - (time.time() - frame_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(
                    f"Frame processing took too long: {time.time() - frame_start - 1.0 / self.config.fps:.3f} seconds too long i.e. {1.0 / (time.time() - frame_start):.2f} FPS. "
                    "Consider decreasing the FPS or optimizing the data function."
                )

            logger.debug(f"Finished sleeping for {sleep_time:.3f} seconds.")

        self.queue.put({"type": "STOP_EPISODE"})
        logger.debug("Finished recording...")

        if on_end:
            on_end()

        self._handle_post_episode()

    def _wait_for_start_signal(self) -> None:
        """Wait until the recording state is set to 'recording'."""
        logger.info("Waiting to start recording...")
        while self.state != "recording":
            if self.state == "exit":
                raise StopIteration
            time.sleep(0.05)

    def _handle_post_episode(self) -> None:
        """Handle the state after recording an episode."""
        if self.state == "paused":
            logger.info("Paused. Awaiting user decision to save/delete...")
            while self.state == "paused":
                time.sleep(0.5)

        if self.state == "to_be_saved":
            logger.info("Saving current episode.")
            self.queue.put({"type": "SAVE_EPISODE"})
            self.episode_count += 1
            self._set_to_wait()
        elif self.state == "to_be_deleted":
            logger.info("Deleting current episode.")
            self.queue.put({"type": "DELETE_EPISODE"})
            self._set_to_wait()
        elif self.state == "exit":
            pass
        else:
            logger.warning(f"Unexpected state after recording: {self.state}")

    def __enter__(self) -> "RecordingManager":  # noqa: D105
        """Enter the recording manager context."""
        print(Panel(self.get_instructions()))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001, D105
        """Exit the recording manager."""
        if exc_type is not None:
            logger.error(
                "An error occurred during recording. Shutting down the recording manager.",
                exc_info=(exc_type, exc_value, traceback),
            )

        if not self.config.push_to_hub:
            logger.info("Not pushing dataset to Hugging Face Hub.")
        else:
            self.queue.put({"type": "PUSH_TO_HUB"})
        logger.info("Shutting down the record process...")
        self.queue.put({"type": "SHUTDOWN"})

        self.writer.join()

    def _set_to_wait(self) -> None:
        """Set to wait if possible."""
        if self.state not in ["to_be_saved", "to_be_deleted"]:
            raise ValueError("Can not go to wait state if the state is not to be saved or deleted!")
        if self.episode_count >= self.config.num_episodes:
            self.state = "exit"
        else:
            self.state = "is_waiting"


class ROSRecordingManager(RecordingManager):
    """ROS-based recording manager for controlling episode recording."""

    def __init__(self, config: RecordingManagerConfig | None = None, **kwargs) -> None:  # noqa: ANN003
        """Initialize ROS recording manager.

        Args:
            config: RecordingManagerConfig instance. If provided, **kwargs are ignored except for backwards compatibility.
            **kwargs: Individual parameters for backwards compatibility.
        """
        super().__init__(config=config, **kwargs)
        self.allowed_actions = ["record", "save", "delete", "exit"]
        self._subscriber = self.node.create_subscription(
            String, "record_transition", self._callback_recording_trigger, 10
        )
        logger.debug("ROS Recording subscriber initialized.")

    @override
    def get_instructions(self) -> str:
        """Returns the instructions to use the recording manager."""
        return (
            "[b]Published messages for recording state:[/b]\n"
            "<record> to start/stop recording.\n"
            "<save> to save the current recorded episode.\n"
            "<delete> to delete the current episode.\n"
            "<exit> to exit the recording manager."
        )

    def _callback_recording_trigger(self, msg: String) -> None:
        """Callback for recording state trigger.

        Args:
            msg: The message containing the recording state
        """
        if msg.data not in self.allowed_actions:
            print(f"[red]Invalid action received: {msg.data}[/red]")
            print("[yellow]Allowed actions are: record, save, delete, exit[/yellow]")
            return

        logger.debug(f"Received message: {msg.data}")
        logger.debug(f"Current state: {self.state}")

        if self.state == "is_waiting":
            if msg.data == "record":
                logger.debug("Transitioning to recording state.")
                if self.bag_recorder:
                    self.bag_recorder.start_episode_recording(self.episode_count)
                self.state = "recording"
            if msg.data == "exit":
                logger.debug("Transitioning to exit state.")
                self.state = "exit"
        elif self.state == "recording":
            if msg.data == "record":
                logger.debug("Transitioning to paused state.")
                if self.bag_recorder:
                    self.bag_recorder.stop_episode_recording()
                self.state = "paused"
        elif self.state == "paused":
            if msg.data == "exit":
                logger.debug("Transitioning to exit state.")
                self.state = "exit"
            if msg.data == "save":
                logger.debug("Transitioning to to_be_saved state.")
                self.state = "to_be_saved"
            if msg.data == "delete":
                logger.debug("Transitioning to to_be_deleted state.")
                self.state = "to_be_deleted"


class KeyboardRecordingManager(RecordingManager):
    """Keyboard-based recording manager for controlling episode recording."""

    def __init__(self, config: RecordingManagerConfig | None = None, **kwargs) -> None:  # noqa: ANN003
        """Initialize keyboard recording manager.

        Args:
            config: RecordingManagerConfig instance. If provided, **kwargs are ignored except for backwards compatibility.
            **kwargs: Individual parameters for backwards compatibility.
        """
        super().__init__(config=config, **kwargs)
        self.listener = keyboard.Listener(on_press=self._on_press)

    @override
    def get_instructions(self) -> str:
        """Returns the instructions to use the recording manager."""
        return "[b]Keys for recording:[/b]\n<r> To start/stop [b]R[/b]ecording.\n<s> To [b]S[/b]ave the current recorded episode.\n<d> to [b]D[/b]elete the current episode.\n<q> To [b]Q[/b]uit the recording."

    def _on_press(self, key: keyboard.KeyCode | keyboard.Key | None) -> None:
        """Handle keyboard press events.

        Args:
            key: The keyboard key that was pressed
        """
        if key is None:
            return

        if isinstance(key, keyboard.Key):
            return

        try:
            if self.state == "is_waiting":
                if key.char == "r":
                    self.state = "recording"
                if key.char == "q":
                    self.state = "exit"
            elif self.state == "recording":
                if key.char == "r":
                    self.state = "paused"
            elif self.state == "paused":
                if key.char == "q":
                    self.state = "exit"
                if key.char == "s":
                    self.state = "to_be_saved"
                if key.char == "d":
                    self.state = "to_be_deleted"
        except AttributeError:
            pass

    def stop(self) -> None:
        """Stop the keyboard listener."""
        self.listener.stop()

    def __enter__(self) -> "RecordingManager":  # noqa: D105
        self.listener.start()
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001, D105
        self.listener.stop()
        super().__exit__(exc_type, exc_value, traceback)


def make_recording_manager(
    recording_manager_type: Literal["keyboard", "ros"],
    config: RecordingManagerConfig | None = None,
    config_path: Path | str | None = None,
    topics_to_record: list[str] | None = None,
    **kwargs: dict,
) -> RecordingManager:
    """Factory function to create a recording manager.

    Args:
        recording_manager_type: Type of recording manager to create.
        config: RecordingManagerConfig instance. Takes precedence over config_path.
        config_path: Path to YAML config file to load.
        topics_to_record: List of ROS2 topic names to record if bag recording is enabled.
        **kwargs: Additional arguments to override config values or for backwards compatibility.

    Returns:
        A RecordingManager instance of the specified type.
    """
    if config is not None:
        if kwargs:
            config_dict = config.__dict__.copy()
            config_dict.update(kwargs)
            final_config = RecordingManagerConfig(**config_dict)
        else:
            final_config = config
    elif config_path is not None:
        final_config = RecordingManagerConfig.from_yaml(config_path, **kwargs)
    else:
        final_config = None

    if recording_manager_type == "keyboard":
        return KeyboardRecordingManager(
            config=final_config, topics_to_record=topics_to_record, **kwargs
        )
    elif recording_manager_type == "ros":
        return ROSRecordingManager(config=final_config, topics_to_record=topics_to_record, **kwargs)
    else:
        raise ValueError(f"Unknown recording manager type: {recording_manager_type}")
