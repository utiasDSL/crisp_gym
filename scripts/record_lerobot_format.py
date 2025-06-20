"""Script showcasing how to record data in Lerobot Format."""

import shutil
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.utils import build_dataset_frame
from pynput import keyboard

from crisp_gym.manipulator_env import ManipulatorCartesianEnv
from crisp_gym.manipulator_env_config import NoCamFrankaEnvConfig

# %%


class RecordingManager:
    """Keyboard event listener for controlling episode recording."""

    def __init__(self, num_episodes: int = 3) -> None:
        """Initialize keyboard listener with state flags."""
        self.state: Literal[
            "is_waiting", "recording", "paused", "to_be_saved", "to_be_deleted", "exit"
        ] = "is_waiting"
        self.listener = keyboard.Listener(on_press=self._on_press)
        self._num_episodes = num_episodes
        self.episode_count = 0

    def _on_press(self, key: Optional[keyboard.Key | keyboard.KeyCode]) -> None:
        """Handle keyboard press events.

        Args:
            key: The keyboard key that was pressed
        """
        try:
            if self.state == "is_waiting":
                if key == keyboard.Key.space:
                    self.state = "recording"
                if key == keyboard.Key.esc:
                    self.state = "exit"
            elif self.state == "recording":
                if key == keyboard.Key.space:
                    self.state = "paused"
            elif self.state == "paused":
                if key == keyboard.Key.space:
                    self.state = "recording"
                if key == keyboard.Key.esc:
                    self.state = "exit"
                if key == keyboard.Key.enter:
                    self.state = "to_be_saved"
                if key == keyboard.Key.delete:
                    self.state = "to_be_deleted"
        except AttributeError:
            pass

    def set_to_wait(self) -> None:
        """Set to wait if possible."""
        if self.state not in ["to_be_saved", "to_be_deleted"]:
            raise ValueError("Can not go to wait state if the state is not to be saved or deleted!")
        self.state = "is_waiting"

    def stop(self) -> None:
        """Stop the keyboard listener."""
        self.listener.stop()

    def not_done(self) -> bool:
        """Return true if we are not done recording."""
        return self.episode_count < self._num_episodes and not self.state == "exit"

    def __enter__(self) -> "RecordingManager":  # noqa: D105
        self.listener.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001, D105
        self.listener.stop()


# %%

# Configuration
cameras = {
    "third_person_image": (256, 256, 3),
    "wrist_image": (256, 256, 3),
}
camera_key_map = {"third_person_image": "primary", "wrist_image": "wrist"}

joint_dims = [f"joint_{idx}" for idx in range(7)] + ["gripper"]
cartesian_dims = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

features = {}
repo_id = "franka_single"
single_task = "pick the lego block."
robot_type = "franka"
fps = 30
max_episode_duration = 30.0  # seconds

# Feature configuration
features["observation.state"] = {
    "dtype": "float32",
    "shape": (len(cartesian_dims),),
    "names": cartesian_dims,
}

features["action"] = {
    "dtype": "float32",
    "shape": (len(cartesian_dims),),
    "names": cartesian_dims,
}

# State dictionaries
obs_dict = {dim: 0.0 for dim in cartesian_dims}
action_dict = {dim: 0.0 for dim in cartesian_dims}

# %%


def create_dataset() -> LeRobotDataset:
    """Create a new dataset instance.

    Returns:
        A new LeRobotDataset instance
    """
    # Clean up existing dataset if it exists
    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=False,
    )


# %%

env_config = NoCamFrankaEnvConfig()
env = ManipulatorCartesianEnv(config=env_config)
print(env._get_obs())

# %%


def record_episodes(num_episodes: int = 3) -> None:
    """Record episodes with keyboard control.

    Args:
        num_episodes: Number of episodes to record
    """
    dataset = create_dataset()

    with RecordingManager(num_episodes=num_episodes) as recording_manager:
        while recording_manager.not_done:
            if recording_manager.state == "is_waiting":
                print("Waiting for user to start.")
            while recording_manager.state == "is_waiting":
                time.sleep(1.0)

            print("Started episode")
            while recording_manager.state == "recording":
                obs = env._get_obs()
                action, _, _, _, _ = env.step(np.array([0.0] * 7))

                action_dict = {dim: action[i] for i, dim in enumerate(cartesian_dims)}
                obs_dict = {
                    dim: obs["cartesian"][i] if i < 6 else obs["gripper"]
                    for i, dim in enumerate(cartesian_dims)
                }

                action_frame = build_dataset_frame(features, action_dict, prefix="action")
                obs_frame = build_dataset_frame(features, obs_dict, prefix="observation.state")
                frame = {**obs_frame, **action_frame}
                dataset.add_frame(frame, task=single_task)
                # time.sleep(1 / fps)

            if recording_manager.state == "paused":
                print("Waiting for user to decide whether to save or delete the episode")
            while recording_manager.state == "paused":
                time.sleep(1.0)

            if recording_manager.state == "to_be_saved":
                print("Saving episode.")
                dataset.save_episode()
                recording_manager.episode_count += 1
                recording_manager.set_to_wait()

            if recording_manager.state == "to_be_deleted":
                print("Deleting episode")
                dataset.clear_episode_buffer()
                recording_manager.set_to_wait()


if __name__ == "__main__":
    record_episodes()
