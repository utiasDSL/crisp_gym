from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Path to your local dataset root

local_dataset_path = "/home/johannes/.cache/huggingface/lerobot/JohannesSaut/insert_lego" 

# The repo id you want to push to (format: "username/dataset_name")

repo_id = "JohannesSaut/insert_lego"



# Load the dataset

dataset = LeRobotDataset(repo_id=repo_id, root=local_dataset_path)



# Push to the hub

dataset.push_to_hub(

    private=False,  # Set to True if you want a private repo

    tags=["LeRobot", "test"],  # Optional: add tags

)