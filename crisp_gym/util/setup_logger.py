"""Setup logging for the project scripts."""

import atexit
import logging
import queue
from logging.handlers import QueueHandler, QueueListener

from rich.logging import RichHandler


def setup_logging(level=logging.INFO):  # noqa: ANN001
    """Recommended logging setup for the project."""

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

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(console_handler)

    handlers = [
        console_handler,
    ]

    listener = QueueListener(log_queue, *handlers)
    listener.start()

    atexit.register(listener.stop)
