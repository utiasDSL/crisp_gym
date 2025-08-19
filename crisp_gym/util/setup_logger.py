"""Setup logging for the project scripts."""

import logging

from rich.logging import RichHandler


def setup_logging(level=logging.INFO):  # noqa: ANN001
    """Recommended logging setup for the project."""
    import logging

    handler = RichHandler(rich_tracebacks=True)
    formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
