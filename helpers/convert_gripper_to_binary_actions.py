"""Helper script to convert gripper actions to binary commands based on gripper state."""

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import einops
from rich import print
from rich.progress import Progress

# Configuration
SOURCE_DATASET = "LSY-lab/drawer_v2"
GRIPPER_THRESHOLD = 0.3  # If gripper state < threshold, set action to 1.0 (close), else 0.0 (open)
TESTING_MODE = True  # Set to False to process all episodes
TEST_EPISODES_LIMIT = 3  # Only process first N episodes when testing

# Load source dataset
dataset = LeRobotDataset(repo_id=SOURCE_DATASET)

print(f"Source dataset: {SOURCE_DATASET}")
print(f"Total episodes: {dataset.meta.total_episodes}")
print(f"Gripper threshold: {GRIPPER_THRESHOLD}")
print(f"Testing mode: {TESTING_MODE}")
if TESTING_MODE:
    print(f"Will process only first {TEST_EPISODES_LIMIT} episodes")

# Verify that observation.state.gripper exists
if "observation.state.gripper" not in dataset.features:
    raise ValueError("Dataset does not contain 'observation.state.gripper' feature.")
else:
    print("\nGripper state information found in the dataset.")

# Keep all features unchanged
features = {}
for key, value in dataset.features.items():
    if key.startswith("observation.") or key == "action":
        features[key] = value

print("\nFeatures for the new dataset:")
for key in features.keys():
    print(f"- {key}")

# Create new dataset name - replace version suffix with _v3
new_repo_id = SOURCE_DATASET.rsplit("_v", 1)[0] + "_v3"
print(f"\nNew dataset repo_id: {new_repo_id}")

# Create new dataset
new_dataset = LeRobotDataset.create(
    repo_id=new_repo_id,
    features=features,
    fps=dataset.fps,
)

# Copy frames with binary gripper actions
with Progress() as progress:
    progress_task = progress.add_task("Converting dataset...", total=dataset.meta.total_episodes)
    current_episode_index = 0

    for frame in dataset:
        new_frame = {}

        # Testing mode: stop after processing specified number of episodes
        if TESTING_MODE and current_episode_index >= TEST_EPISODES_LIMIT:
            break

        for key in features.keys():
            if key == "action":
                # Convert gripper action to binary based on gripper state
                action = frame[key].copy()
                gripper_state = frame["observation.state.gripper"]

                # Extract gripper state value (handle both scalar and array)
                if hasattr(gripper_state, "shape") and gripper_state.shape:
                    gripper_value = gripper_state.item()
                else:
                    gripper_value = float(gripper_state)

                # Set binary gripper action based on threshold
                if gripper_value < GRIPPER_THRESHOLD:
                    action[6] = 1.0  # Close gripper
                else:
                    action[6] = 0.0  # Open gripper

                new_frame[key] = action.astype(np.float32)
            elif key.startswith("observation.images"):
                image = einops.rearrange(frame[key], "c h w -> h w c")
                new_frame[key] = image
            elif key.startswith("observation.state"):
                if not frame[key].shape:
                    new_frame[key] = np.array([frame[key]], dtype=np.float32)
                    continue
                new_frame[key] = frame[key].squeeze()

        if frame["episode_index"] > current_episode_index:
            current_episode_index = frame["episode_index"]
            progress.update(progress_task, advance=1)
            new_dataset.save_episode()

        new_dataset.add_frame(new_frame, task=frame.get("task", ""))

    new_dataset.save_episode()
    progress.update(progress_task, advance=1)

print("\nConversion completed.")
if TESTING_MODE:
    print(f"Processed {TEST_EPISODES_LIMIT} episodes in testing mode.")
    print("Set TESTING_MODE = False to process all episodes.")
else:
    print("Pushing to hub...")
    new_dataset.push_to_hub()
