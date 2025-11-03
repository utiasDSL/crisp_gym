"""A script to check and print all available pre-trained policies in the lerobot library."""

from lerobot.configs.train import PreTrainedConfig
from rich import print

print(f"The available policies are: {PreTrainedConfig.get_known_choices().keys()}")
