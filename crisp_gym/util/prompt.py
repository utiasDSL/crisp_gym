"""Prompt utility for user input in command-line applications."""

import logging

logger = logging.getLogger(__name__)


def prompt(
    message: str = "Choose an option:",
    options: list | None = None,
    default: str | None = None,
) -> str:
    """Prompt the user to choose from a list of options or just enter a string.

    Args:
        message (str): The prompt message to display.
        options (list, optional): A list of string options to choose from.
        default (str, optional): The default value to use if user enters nothing.

    Returns:
        str: The selected or entered string.
    """
    logger.info("-" * 40)
    if options:
        logger.info(message)
        for i, option in enumerate(options, 1):
            logger.info(f"{i}. {option}")
        if default:
            logger.info(f"(Default: {default})")
        logger.info("-" * 40)

        while True:
            logger.info("Enter number, text, or press Enter for default: ")
            choice = input().strip()
            if not choice:
                if default:
                    return default
                else:
                    logger.info("No input given and no default set. Try again.")
                    continue
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(options):
                    return options[index]
                else:
                    logger.info("Invalid number. Try again.")
            elif choice in options:
                return choice
            else:
                logger.info("Invalid input. Try again.")
    else:
        while True:
            if default is not None:
                message += f" (Default: '{default}')"
            logger.info(message)
            logger.info("-" * 40)
            logger.info("Enter string or press Enter for default: ")
            response = input().strip()
            if response:
                return response
            elif default is not None:
                return default
            else:
                logger.info("No input given and no default set. Try again.")
