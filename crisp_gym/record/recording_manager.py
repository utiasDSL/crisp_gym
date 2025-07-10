"""Keyboard event listener for controlling episode recording."""

import logging
import multiprocessing as mp
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import rclpy
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame
from pynput import keyboard
from rclpy.executors import SingleThreadedExecutor
from rich import print
from rich.logging import RichHandler
from rich.panel import Panel
from std_msgs.msg import String
from typing_extensions import override

FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])


class RecordingManager(ABC):
    """Base class for event listener to control episode recording."""

    def __init__(
        self,
        features: dict,
        repo_id: str,
        task: str,
        robot_type: str = "Franka",
        resume: bool = False,
        fps: int = 30,
        num_episodes: int = 3,
    ) -> None:
        """Initialize the recording manager.

        Args:
            features: The features to record.
            repo_id: The repository ID for the dataset.
            task: The task for which the dataset is being recorded.
            robot_type: The type of robot (default is "Franka").
            resume: Whether to resume from an existing dataset (default is False).
            fps: Frames per second for the dataset (default is 30).
            num_episodes: Number of episodes to record (default is 3).

        """
        self.state: Literal[
            "is_waiting", "recording", "paused", "to_be_saved", "to_be_deleted", "exit"
        ] = "is_waiting"

        self.features = features
        self.repo_id = repo_id
        self.task = task
        self.robot_type = robot_type
        self.resume = resume
        self.fps = fps
        self.num_episodes = num_episodes
        self.episode_count = 0

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

    def _create_dataset(self) -> LeRobotDataset:
        """Factory function to create a dataset object."""
        logging.info("Creating dataset object.")
        if self.resume:
            logging.info(
                f"[green]Resuming recording from existing dataset: {self.repo_id}",
                extra={"markup": True},
            )
            dataset = LeRobotDataset(repo_id=self.repo_id)
            if self.num_episodes <= dataset.num_episodes:
                logging.error(
                    f"The dataset already has {dataset.num_episodes} recorded. Please select a larger number."
                )
                exit()
            logging.info(
                f"Resuming from episode {dataset.num_episodes} with {self.num_episodes} episodes to record."
            )
            self.episode_count_queue.put(dataset.num_episodes - 1)
        else:
            logging.info(f"[green]Creating new dataset: {self.repo_id}", extra={"markup": True})
            # Clean up existing dataset if it exists
            if Path(HF_LEROBOT_HOME / self.repo_id).exists():
                logging.error(
                    f"The repo_id already exists. If you intended to resume the collection of data, then execute this script with the --resume flag. Otherwise remove it:\n'rm -r {str(Path(HF_LEROBOT_HOME / self.repo_id))}'."
                )
                exit()
            dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                robot_type=self.robot_type,
                features=self.features,
                use_videos=False,
            )
        return dataset

    def _writer_proc(self):
        """Process to write data to the dataset."""
        logging.info("Starting dataset writer process.")
        dataset = self._create_dataset()
        self.dataset_ready.set()
        camera_names = [
            name.split(".")[-1] for name in self.features if name.startswith("observation.images.")
        ]

        while True:
            msg = self.queue.get()
            try:
                mtype = msg["type"]

                if mtype == "FRAME":
                    obs, action = msg["data"]

                    logging.debug(f"Received frame with action: {action} and obs: {obs}")

                    action_dict = {
                        dim: action[i] for i, dim in enumerate(self.features["action"]["names"])
                    }

                    obs_dict = {
                        dim: (obs["cartesian"][i] if i < 6 else obs["gripper"][0])
                        for i, dim in enumerate(self.features["observation.state"]["names"])
                    }
                    cam_frame = {
                        f"observation.images.{camera_name}": obs[f"{camera_name}_image"]
                        for camera_name in camera_names
                    }

                    logging.debug(f"Action dict: {action_dict}")
                    logging.debug(f"Observation dict: {obs_dict}")

                    action_frame = build_dataset_frame(self.features, action_dict, prefix="action")
                    obs_frame = build_dataset_frame(
                        self.features, obs_dict, prefix="observation.state"
                    )

                    frame = {
                        **obs_frame,
                        **action_frame,
                        **cam_frame,
                    }

                    dataset.add_frame(frame, task=self.task)

                elif mtype == "SAVE_EPISODE":
                    dataset.save_episode()

                elif mtype == "DELETE_EPISODE":
                    dataset.clear_episode_buffer()

                elif mtype == "PUSH_TO_HUB":
                    logging.info(
                        "[green]Pushing dataset to Hugging Face Hub...", extra={"markup": True}
                    )
                    dataset.push_to_hub(repo_id=self.repo_id, private=True)
                    logging.info(
                        "[green]Dataset pushed to Hugging Face Hub successfully.",
                        extra={"markup": True},
                    )

                elif mtype == "SHUTDOWN":
                    break
            finally:
                self.queue.task_done()

    def done(self) -> bool:
        """Return true if we are done recording."""
        return self.state == "exit"

    def record(self, obs: dict, action: dict) -> None:
        """Record a frame with the given observation and action.

        Args:
            obs: The observation dictionary.
            action: The action dictionary.
        """
        if self.state != "recording":
            raise ValueError("Can not record if the state is not recording!")
        self.queue.put({"type": "FRAME", "data": (obs, action)})

    def delete_episode(self) -> None:
        """Delete the current episode."""
        if self.state != "to_be_deleted":
            raise ValueError("Can not save episode if the state is not to be deleted!")
        logging.info("[red]Deleting current episode.", extra={"markup": True})

        self.queue.put({"type": "DELETE_EPISODE"})
        self.set_to_wait()

    def save_episode(self) -> None:
        """Save the current episode."""
        if self.state != "to_be_saved":
            raise ValueError("Can not save episode if the state is not paused!")

        logging.info("[green]Saving current episode.", extra={"markup": True})

        self.queue.put({"type": "SAVE_EPISODE"})
        self.episode_count += 1
        self.set_to_wait()

    def push_to_hub(self) -> None:
        """Push the dataset to the Hugging Face Hub."""
        if self.state != "exit":
            raise ValueError("Can not push to hub if the state is not exit!")
        self.queue.put({"type": "PUSH_TO_HUB"})

    def shutdown(self) -> None:
        """Shutdown the recording manager."""
        logging.info("[red]Shutting down recording manager.", extra={"markup": True})
        self.queue.put({"type": "SHUTDOWN"})
        self.writer.join()

    @abstractmethod
    def get_instructions(self) -> str:
        """Return the instructions to use the recording manager."""
        raise NotImplementedError()

    def __enter__(self) -> "RecordingManager":  # noqa: D105
        """Enter the recording manager context."""
        print(Panel(self.get_instructions()))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001, D105
        """Exit the recording manager."""
        pass

    def set_to_wait(self) -> None:
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

        if self.state == "is_waiting":
            if msg.data == "record":
                self.state = "recording"
            if msg.data == "exit":
                self.state = "exit"
        elif self.state == "recording":
            if msg.data == "record":
                self.state = "paused"
        elif self.state == "paused":
            if msg.data == "exit":
                self.state = "exit"
            if msg.data == "save":
                self.state = "to_be_saved"
            if msg.data == "delete":
                self.state = "to_be_deleted"


class KeyboardRecordingManager(RecordingManager):
    """Keyboard event listener for controlling episode recording."""

    def __init__(self, **kwargs: dict) -> None:
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
