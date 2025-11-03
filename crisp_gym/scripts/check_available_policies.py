from lerobot.configs.train import PreTrainedConfig
from rich import print

print(f"The available policies are: {PreTrainedConfig.get_known_choices().keys()}")
