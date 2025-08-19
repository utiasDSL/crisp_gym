"""Custom logger with rich handler support."""

import logging

from rich.logging import RichHandler


def setup_logging(level=logging.INFO):
    """Recommended logging setup for the project."""
    import logging

    handler = RichHandler()
    formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
    handler.setFormatter(formatter)

    handlers = [RichHandler(rich_tracebacks=True)]
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
