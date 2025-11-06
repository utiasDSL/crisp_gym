"""Simple class to evaluate the performance of a model after an episode.

Usage:

```python
evaluator = Evaluator(output_file="evaluation_results.csv")

with evaluator.start_eval():  # Opens the output file for writing results
    with recording_manager:  # Starts the recording manager to manage episodes and save records
        while not recording_manager.done():
            recording_manager.record_episode(...)  # Record episode using policy
            evaluator.evaluate(
                episode=recording_manager.episode_count
            )  # Let the evaluator evaluate the episode (by asking the user for input)
```
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

from crisp_gym.util.prompt import prompt
from crisp_gym.util.setup_logger import setup_logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class to record and evaluate the performance of a model in a gym environment."""

    def __init__(self, output_file: str | Path):
        """Initialize the Evaluator with the environment and output file.

        Args:
            output_file (str | Path): The path to the output file where results will be saved. If a string is provided, then it will be saved in outputs/<output_file>.
        """
        self.output_file: Path = (
            Path("outputs") / output_file if isinstance(output_file, str) else output_file
        )
        self.eval_writer = None
        logger.info(f"Evaluator initialized. Results will be saved to {self.output_file}.")

        self.start_time = None

    @contextmanager
    def start_eval(self, overwrite: bool = True, activate: bool = True):
        """Context manager to handle the evaluation process."""
        if not activate:
            logger.info("Evaluation is deactivated. No results will be saved.")
            yield
            return

        if overwrite and self.output_file.exists():
            with self.output_file.open("w") as f:
                f.write("episode,success,score,time\n")

        self.eval_writer = self.output_file.open("a")

        try:
            yield
        finally:
            self.eval_writer.close()

    def start_timer(self):
        """Start the evaluation timer."""
        self.start_time = time.time()

    def evaluate(self, episode: int):
        """Evaluate the performance of the model after an episode using user input.

        We only evaluate the episode if the eval_writer is set, which means we are in evaluation mode.

        Args:
            episode (int): The episode number to evaluate.
        """
        if self.start_time is None:
            logger.warning(
                "Evaluation timer was not started. Not recording time taken for evaluation."
            )
            elapsed_time = None
        else:
            end_time = time.time()
            elapsed_time = end_time - self.start_time
            logger.info(f"Episode {episode} took {elapsed_time:.2f} seconds.")
            self.start_time = None

        if self.eval_writer:
            logger.info(f"Evaluating episode {episode}. Please provide your feedback.")
            success = prompt(
                message="Was the episode successful? (yes/no)", options=["yes", "no"], default="yes"
            )
            score = prompt(
                message="Please provide a score/completion progress for the episode (0-10):",
                default="5",
            )

            score = int(score)
            if not (0 <= score <= 10):
                raise ValueError("Score must be between 0 and 10.")

            self.eval_writer.write(f"{episode},{success},{score},{elapsed_time:.2f}\n")
            self.eval_writer.flush()
            logger.info(
                f"Recorded evaluation for episode {episode}: success={success}, score={score}, time={elapsed_time:.2f} seconds."
            )
        else:
            logger.debug("Evaluation writer is not set. Skipping evaluation.")


if __name__ == "__main__":
    import random

    setup_logging(level=logging.INFO)
    evaluator = Evaluator("evaluation_results.csv")
    with evaluator.start_eval():
        for episode in range(1, 6):  # Simulating 5 episodes
            evaluator.start_timer()
            time.sleep(random.uniform(1.0, 3.0))
            evaluator.evaluate(episode)
    logger.info("Evaluation completed and results saved to evaluation_results.csv.")
