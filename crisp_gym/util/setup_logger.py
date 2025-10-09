"""Setup logging for the project scripts."""

import atexit
import logging
import queue
from logging.handlers import QueueHandler, QueueListener

from rich.logging import RichHandler


def setup_logging(level=logging.INFO, output_to_console: bool = True, output_to_file: bool = True):  # noqa: ANN001
    """Recommended logging setup for the project."""
    import logging

    assert output_to_console or output_to_file, (
        "At least one output (console or file) must be enabled."
    )

    console_formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setFormatter(console_formatter)

    file_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]"
    )
    file_handler = logging.FileHandler("crisp.log")
    file_handler.setFormatter(file_formatter)

    log_queue = queue.Queue()
    queue_handler = QueueHandler(log_queue)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(queue_handler)

    handlers = []
    if output_to_console:
        handlers.append(console_handler)
    if output_to_file:
        handlers.append(file_handler)

    listener = QueueListener(log_queue, *handlers)
    listener.start()

    atexit.register(listener.stop)
