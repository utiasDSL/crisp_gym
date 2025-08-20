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

    @contextmanager
    def start_eval(self, overwrite: bool = True, activate: bool = True):
        """Context manager to handle the evaluation process."""
        if not activate:
            logger.info("Evaluation is deactivated. No results will be saved.")
            yield
            return

        if overwrite and self.output_file.exists():
            with self.output_file.open("w") as f:
                f.write("episode,success,score\n")

        self.eval_writer = self.output_file.open("a")

        try:
            yield
        finally:
            self.eval_writer.close()

    def evaluate(self, episode: int):
        """Evaluate the performance of the model after an episode using user input.

        We only evaluate the episode if the eval_writer is set, which means we are in evaluation mode.

        Args:
            episode (int): The episode number to evaluate.
        """
        if self.eval_writer:
            logger.info(f"Evaluating episode {episode}. Please provide your feedback.")
            success = prompt(
                message="Was the episode successful? (yes/no)", options=["yes", "no"], default="yes"
            )
            score = prompt(message="Please provide a score for the episode (0-10):", default="5")

            try:
                score = int(score)
                if not (0 <= score <= 10):
                    raise ValueError("Score must be between 0 and 10.")
            except ValueError as e:
                logger.error(f"Invalid score input: {e}. Defaulting to 5.")
                score = 5

                self.eval_writer.write(f"{episode},{success},{score}\n")
                self.eval_writer.flush()
                logger.info(
                    f"Recorded evaluation for episode {episode}: success={success}, score={score}"
                )
        else:
            logger.debug("Evaluation writer is not set. Skipping evaluation.")


if __name__ == "__main__":
    setup_logging(level=logging.INFO)
    evaluator = Evaluator("evaluation_results.csv")
    with evaluator.start_eval():
        for episode in range(1, 6):  # Simulating 5 episodes
            evaluator.evaluate(episode)
    logger.info("Evaluation completed and results saved.")
