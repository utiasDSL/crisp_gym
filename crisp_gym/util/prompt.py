"""Prompt utility for user input in command-line applications."""

import logging


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
    if options:
        logging.info(message)
        for i, option in enumerate(options, 1):
            logging.info(f"{i}. {option}")
        if default:
            logging.info(f"(Default: {default})")

        while True:
            choice = input("Enter number, text, or press Enter for default: ").strip()
            if not choice:
                if default:
                    return default
                else:
                    logging.info("No input given and no default set. Try again.")
                    continue
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(options):
                    return options[index]
                else:
                    logging.info("Invalid number. Try again.")
            elif choice in options:
                return choice
            else:
                logging.info("Invalid input. Try again.")
    else:
        while True:
            if default is not None:
                message += f" (Default: '{default}')"
            logging.info(message)
            response = input("Enter string or press Enter for default: ").strip()
            if response:
                return response
            elif default is not None:
                return default
            else:
                logging.info("No input given and no default set. Try again.")
