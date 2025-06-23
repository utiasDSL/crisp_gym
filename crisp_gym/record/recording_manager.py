"""Keyboard event listener for controlling episode recording."""

from typing import Literal

from pynput import keyboard
from rich import print
from rich.panel import Panel


class RecordingManager:
    """Keyboard event listener for controlling episode recording."""

    def __init__(self, num_episodes: int = 3) -> None:
        """Initialize keyboard listener with state flags."""
        self.state: Literal[
            "is_waiting", "recording", "paused", "to_be_saved", "to_be_deleted", "exit"
        ] = "is_waiting"
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.num_episodes = num_episodes
        self.episode_count = 0

    def get_instructions(self) -> str:
        """Returns the instructions to use the recording manager."""
        return "[b]Keys for recording:[/b]\n<r> To start/stop [b]R[/b]ecording.\n<s> To [b]S[/b]ave the current recorded episode.\n<d> to [b]D[/b]elete the current episode.\n<q> To [b]Q[/b]uit the recording."

    def _on_press(self, key: keyboard.KeyCode) -> None:
        """Handle keyboard press events.

        Args:
            key: The keyboard key that was pressed
        """
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

    def set_to_wait(self) -> None:
        """Set to wait if possible."""
        if self.state not in ["to_be_saved", "to_be_deleted"]:
            raise ValueError("Can not go to wait state if the state is not to be saved or deleted!")
        if self.episode_count >= self.num_episodes:
            self.state = "exit"
        else:
            self.state = "is_waiting"

    def stop(self) -> None:
        """Stop the keyboard listener."""
        self.listener.stop()

    def done(self) -> bool:
        """Return true if we are done recording."""
        return self.state == "exit"

    def __enter__(self) -> "RecordingManager":  # noqa: D105
        self.listener.start()
        print(Panel(self.get_instructions()))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001, D105
        self.listener.stop()
