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
from std_msgs.msg import String
from typing_extensions import override

from crisp_gym.util.lerobot_features import concatenate_state_features

logger = logging.getLogger(__name__)


class RecordingManager(ABC):
    """Base class for event listener to control episode recording."""

    def __init__(
        self,
        features: dict,
        repo_id: str,
        robot_type: str = "Franka",
        resume: bool = False,
        fps: int = 30,
        num_episodes: int = 3,
        push_to_hub: bool = False,
        use_sound: bool = True,
    ) -> None:
        """Initialize the recording manager.

        Args:
            features: The features to record.
            repo_id: The repository ID for the dataset.
            robot_type: The type of robot (default is "Franka").
            resume: Whether to resume from an existing dataset (default is False).
            fps: Frames per second for the dataset (default is 30).
            num_episodes: Number of episodes to record (default is 3).
            push_to_hub: Whether to push the dataset to Hugging Face Hub (default is False).
            use_sound: Whether to use sound notifications for episode completion (default is True).
        """
        self.state: Literal[
            "is_waiting", "recording", "paused", "to_be_saved", "to_be_deleted", "exit"
        ] = "is_waiting"

        self.features = features
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.resume = resume
        self.fps = fps
        self.num_episodes = num_episodes
        self.push_to_hub = push_to_hub
        self.use_sound = use_sound
        self.episode_count = 0

        # TODO: do not hardcode the queue size, make it configurable
        self.queue = mp.JoinableQueue(16)
        self.episode_count_queue = mp.Queue(1)
        self.dataset_ready = mp.Event()

        # Start the writer process
        self.writer = mp.Process(
            target=self._writer_proc,
            args=(),
            name="dataset_writer",
            daemon=True,
        )
        self.writer.start()

    def wait_until_ready(self) -> None:
        """Wait until the dataset writer is ready."""
        self.dataset_ready.wait()
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

    @abstractmethod
    def get_instructions(self) -> str:
        """Return the instructions to use the recording manager."""
        raise NotImplementedError()

    def _create_dataset(self) -> LeRobotDataset:
        """Factory function to create a dataset object."""
        logger.debug("Creating dataset object.")
        if self.resume:
            logger.info(f"Resuming recording from existing dataset: {self.repo_id}")
            dataset = LeRobotDataset(repo_id=self.repo_id)
            if self.num_episodes <= dataset.num_episodes:
                logger.error(
                    f"The dataset already has {dataset.num_episodes} recorded. Please select a larger number."
                )
                exit()
            logger.info(
                f"Resuming from episode {dataset.num_episodes} with {self.num_episodes} episodes to record."
            )
            self.episode_count_queue.put(dataset.num_episodes - 1)
        else:
            logger.info(f"[green]Creating new dataset: {self.repo_id}", extra={"markup": True})
            # Clean up existing dataset if it exists
            if Path(HF_LEROBOT_HOME / self.repo_id).exists():
                logger.error(
                    f"The repo_id already exists. If you intended to resume the collection of data, then execute this script with the --resume flag. Otherwise remove it:\n'rm -r {str(Path(HF_LEROBOT_HOME / self.repo_id))}'."
                )
                exit()
            dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                robot_type=self.robot_type,
                features=self.features,
                use_videos=True,
            )
            logger.debug(f"Dataset created with meta: {dataset.meta}")
        return dataset

    def _writer_proc(self):
        """Process to write data to the dataset."""
        logger.info("Starting dataset writer process.")
        dataset = self._create_dataset()
        self.dataset_ready.set()
        logger.debug(f"Dataset features: {list(self.features.keys())}")

        while True:
            msg = self.queue.get()
            logger.debug(f"Received message: {msg['type']}")
            try:
                mtype = msg["type"]

                if mtype == "FRAME":
                    obs, action, task = msg["data"]

                    logger.debug(f"Received frame with action: {action} and obs: {obs.keys()}")

                    # Build frame directly from observation using feature-based approach
                    frame = {"action": action.astype(np.float32)}

                    # Add all observation features that match our dataset features
                    for feature_name in self.features:
                        if feature_name == "action":
                            continue  # Already added above
                        if feature_name in obs:
                            value = obs[feature_name]
                            if isinstance(value, np.ndarray) and feature_name.startswith(
                                "observation.state"
                            ):
                                frame[feature_name] = value.astype(np.float32)
                            else:
                                frame[feature_name] = value

                    # Concatenate state vector
                    frame["observation.state"] = concatenate_state_features(obs, self.features)

                    logger.debug(f"Constructed frame with keys: {frame.keys()}")
                    dataset.add_frame(frame, task=task)

                elif mtype == "SAVE_EPISODE":
                    if self.use_sound:
                        try:
                            subprocess.Popen(
                                [
                                    "paplay",
                                    "/usr/share/sounds/freedesktop/stereo/complete.oga ",
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to play sound for episode completion: {e}",
                            )

                    logger.info("Saving current episode to dataset.")
                    # Check if there are frames in the current episode buffer
                    if hasattr(dataset, "_episode_buffer") and len(dataset._episode_buffer) == 0:
                        logger.warning("Episode buffer is empty. No frames to save.")
                    else:
                        logger.info(
                            f"Saving episode with {len(getattr(dataset, '_episode_buffer', []))} frames."
                        )

                    dataset.save_episode()
                    logger.info(
                        f"Episode {self.episode_count} saved to dataset.",
                    )

                elif mtype == "DELETE_EPISODE":
                    if self.use_sound:
                        try:
                            subprocess.Popen(
                                [
                                    "paplay",
                                    "/usr/share/sounds/freedesktop/stereo/suspend-error.oga",
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to play sound for episode deletion: {e}",
                            )

                    dataset.clear_episode_buffer()

                elif mtype == "PUSH_TO_HUB":
                    logger.info(
                        "Pushing dataset to Hugging Face Hub...",
                    )
                    try:
                        dataset.push_to_hub(repo_id=self.repo_id, private=True)
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

        while self.state == "recording":
            frame_start = time.time()

            obs, action = data_fn()

            if obs is None or action is None:
                logger.debug("Data function returned None, skipping frame.")
                # If the data function returns None, skip this frame
                sleep_time = 1 / self.fps - (time.time() - frame_start)
                time.sleep(sleep_time)
                continue

            self.queue.put({"type": "FRAME", "data": (obs, action, task)})

            sleep_time = 1 / self.fps - (time.time() - frame_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logger.warning(
                    f"Frame processing took too long: {time.time() - frame_start - 1.0 / self.fps:.3f} seconds too long i.e. {1.0 / (time.time() - frame_start):.2f} FPS. "
                    "Consider decreasing the FPS or optimizing the data function."
                )
            logger.debug(f"Finished sleeping for {sleep_time:.3f} seconds.")

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

        if not self.push_to_hub:
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
        if self.episode_count >= self.num_episodes:
            self.state = "exit"
        else:
            self.state = "is_waiting"


class ROSRecordingManager(RecordingManager):
    """Keyboard event listener for controlling episode recording."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize keyboard listener with state flags."""
        super().__init__(**kwargs)
        if not rclpy.ok():
            raise RuntimeError(
                "ROS2 is not initialized. Please initialize ROS2 before using the RecordingManager."
            )
        self.allowed_actions = ["record", "save", "delete", "exit"]
        self.node = rclpy.create_node("recording_manager")
        self._subscriber = self.node.create_subscription(
            String, "record_transition", self._callback_recording_trigger, 10
        )
        logger.debug("ROS2 node created and subscriber initialized.")

        threading.Thread(target=self._spin_node, daemon=True).start()

    def _spin_node(self):
        """Spin the ROS2 node in a separate thread."""
        executor = SingleThreadedExecutor()
        executor.add_node(self.node)
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)

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
                self.state = "recording"
            if msg.data == "exit":
                logger.debug("Transitioning to exit state.")
                self.state = "exit"
        elif self.state == "recording":
            if msg.data == "record":
                logger.debug("Transitioning to paused state.")
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
    """Keyboard event listener for controlling episode recording."""

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        """Initialize keyboard listener with state flags."""
        super().__init__(**kwargs)
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
    **kwargs: dict,
) -> RecordingManager:
    """Factory function to create a recording manager."""
    if recording_manager_type == "keyboard":
        return KeyboardRecordingManager(**kwargs)
    elif recording_manager_type == "ros":
        return ROSRecordingManager(**kwargs)
    else:
        raise ValueError(f"Unknown recording manager type: {recording_manager_type}")
