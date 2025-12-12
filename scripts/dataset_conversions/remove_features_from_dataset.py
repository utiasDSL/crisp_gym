"""Helper script to create a new dataset by removing specified features."""

import re

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import einops
from rich import print
from rich.progress import Progress

# Configuration
SOURCE_DATASET = "LSY-lab/drawer_v1"
FEATURES_TO_REMOVE = [
    "observation.state.joints",
    "observation.state.target",
    "observation.state.sensors_ft_sensor",
    "observation.state.sensors_tactile_sensor",
    "observation.images.tactile",
]
WHAT_HAS_BEEN_REMOVED = "ft_tact_target"

# Load source dataset
dataset = LeRobotDataset(repo_id=SOURCE_DATASET)

print(f"Source dataset: {SOURCE_DATASET}")
print(f"Total episodes: {dataset.meta.total_episodes}")
print(f"Features to remove: {FEATURES_TO_REMOVE}")

# Build filtered features
features = {}
observation_state_indices_to_keep = []

# First, collect all features except the ones to remove
for key, value in dataset.features.items():
    if key not in FEATURES_TO_REMOVE:
        if key.startswith("observation.") or key == "action":
            features[key] = value

# Handle observation.state filtering
if "observation.state" in features:
    # Get all names in observation.state
    state_names = dataset.features["observation.state"]["names"]
    print(f"\nOriginal observation.state names: {state_names}")

    # Collect names from features to remove
    names_to_remove = set()
    for feature_to_remove in FEATURES_TO_REMOVE:
        if (
            feature_to_remove.startswith("observation.state.")
            and feature_to_remove in dataset.features
        ):
            sub_feature_names = dataset.features[feature_to_remove]["names"]
            names_to_remove.update(sub_feature_names)

    print(f"Names to remove from observation.state: {names_to_remove}")

    # Build list of indices to keep
    for idx, name in enumerate(state_names):
        if name not in names_to_remove:
            observation_state_indices_to_keep.append(idx)

    print(f"Indices to keep in observation.state: {observation_state_indices_to_keep}")

    # Update the features dict with filtered names and shape
    filtered_state_names = [state_names[i] for i in observation_state_indices_to_keep]
    features["observation.state"]["names"] = filtered_state_names
    features["observation.state"]["shape"] = (len(filtered_state_names),)
    print(f"Filtered observation.state names: {filtered_state_names}")
    print(f"Updated observation.state shape: {features['observation.state']['shape']}")

print("\nFeatures for the new dataset:")
for key in features.keys():
    print(f"- {key}")

# Create new dataset name
# Preserve version suffix if present
new_repo_id = SOURCE_DATASET
version_suffix = ""
version_match = re.search(r"_v\d+$", SOURCE_DATASET)
if version_match:
    version_suffix = version_match.group()
    new_repo_id = SOURCE_DATASET[: version_match.start()]

new_repo_id = f"{new_repo_id}_without_{WHAT_HAS_BEEN_REMOVED}{version_suffix}"
print(f"\nNew dataset repo_id: {new_repo_id}")

# Create new dataset
new_dataset = LeRobotDataset.create(
    repo_id=new_repo_id,
    features=features,
    fps=dataset.fps,
)

# Copy frames with filtering
with Progress() as progress:
    progress_task = progress.add_task("Converting dataset...", total=dataset.meta.total_episodes)
    current_episode_index = 0

    for frame in dataset:
        new_frame = {}


        for key in features.keys():
            if key == "action":
                new_frame[key] = frame[key]
            elif key.startswith("observation.images"):
                image = einops.rearrange(frame[key], "c h w -> h w c")
                new_frame[key] = image
            elif key == "observation.state":
                # Filter the observation.state array
                original_state = frame[key]
                filtered_state = original_state[observation_state_indices_to_keep]

                if not filtered_state.shape:
                    new_frame[key] = np.array([filtered_state], dtype=np.float32)
                else:
                    new_frame[key] = filtered_state.squeeze()
            elif key.startswith("observation.state."):
                # Handle other observation.state.* features that weren't removed
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

print("Conversion completed.")
new_dataset.push_to_hub()
