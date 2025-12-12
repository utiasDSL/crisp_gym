"""Helper script to fix rotation vector discontinuities in an existing dataset.

This script applies the same fix as in manipulator_env.py to ensure:
1. First rotation vector element is positive at episode start
2. No discontinuous jumps between frames (flips vectors pointing in opposite direction)
"""

import re

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import einops
from numpy.typing import NDArray
from rich import print
from rich.progress import Progress

# Configuration
SOURCE_DATASET = "LSY-lab/drawer_v1"
LIMIT_EPISODES = None  # Set to a number to test with limited episodes, None for all


def _point_in_opposite_direction(vector1: NDArray, vector2: NDArray) -> bool:
    """Check if a point is in the opposite direction of a given vector.

    Args:
        vector1 (NDArray): The reference vector.
        vector2 (NDArray): The vector to check.

    Returns:
        bool: True if the vectors point in opposite directions, False otherwise.
    """
    vector1_norm = np.linalg.norm(vector1)
    vector2_norm = np.linalg.norm(vector2)

    if vector1_norm == 0 or vector2_norm == 0:
        return False

    unit_vector1 = vector1 / vector1_norm
    unit_vector2 = vector2 / vector2_norm

    dot_product = np.dot(unit_vector1, unit_vector2)

    return dot_product < 0


def _flip_rotation_vector_if_needed(
    previous_rotation_vector: NDArray | None,
    rotation_vector: NDArray,
) -> NDArray:
    """Flip rotation vector if needed for proper representation.

    Args:
        previous_rotation_vector: Previous rotation vector, or None for first frame
        rotation_vector: Current rotation vector to potentially flip

    Returns:
        Fixed rotation vector
    """
    if previous_rotation_vector is not None:
        if _point_in_opposite_direction(previous_rotation_vector, rotation_vector):
            rotation_vector = -rotation_vector
    else:
        # Make sure that the first element is positive for consistency
        if rotation_vector[0] < 0:
            rotation_vector = -rotation_vector

    return rotation_vector


# Load source dataset
dataset = LeRobotDataset(repo_id=SOURCE_DATASET)

print(f"Source dataset: {SOURCE_DATASET}")
print(f"Total episodes: {dataset.meta.total_episodes}")
print("\nDataset features:")
for key in dataset.features.keys():
    if key.startswith("observation."):
        print(f"  {key}: {dataset.features[key]}")

# Identify rotation vector indices in observation.state
state_names = dataset.features["observation.state"]["names"]
print(f"\nObservation.state names: {state_names}")

# Find indices for cartesian and target rotation vectors
cartesian_rotation_indices = []
target_rotation_indices = []

for idx, name in enumerate(state_names):
    # Assuming cartesian rotation is at indices corresponding to rotation components
    if "cartesian" in name and any(x in name for x in ["rx", "ry", "rz", "rot"]):
        cartesian_rotation_indices.append(idx)
    elif "target" in name and any(x in name for x in ["rx", "ry", "rz", "rot"]):
        target_rotation_indices.append(idx)

# If we can't find by name, we need to use the feature structure
if "observation.state.cartesian" in dataset.features:
    cartesian_names = dataset.features["observation.state.cartesian"]["names"]
    print(f"Cartesian feature names: {cartesian_names}")
    # Rotation is at indices 3:6 in cartesian pose
    cartesian_start_idx = 0
    for idx, name in enumerate(state_names):
        if name == cartesian_names[0]:
            cartesian_start_idx = idx
            break
    cartesian_rotation_indices = [
        cartesian_start_idx + 3,
        cartesian_start_idx + 4,
        cartesian_start_idx + 5,
    ]

if "observation.state.target" in dataset.features:
    target_names = dataset.features["observation.state.target"]["names"]
    print(f"Target feature names: {target_names}")
    # Rotation is at indices 3:6 in target pose
    target_start_idx = 0
    for idx, name in enumerate(state_names):
        if name == target_names[0]:
            target_start_idx = idx
            break
    target_rotation_indices = [target_start_idx + 3, target_start_idx + 4, target_start_idx + 5]

print(f"\nCartesian rotation indices in observation.state: {cartesian_rotation_indices}")
print(f"Target rotation indices in observation.state: {target_rotation_indices}")

# Create new dataset name
new_repo_id = re.sub(r"_v\d+$", "", SOURCE_DATASET) + "_v2"
print(f"\nNew dataset repo_id: {new_repo_id}")

# Create new dataset with same features
new_dataset = LeRobotDataset.create(
    repo_id=new_repo_id,
    features=dataset.features,
    fps=dataset.fps,
)

# Process frames with rotation vector fix
with Progress() as progress:
    progress_task = progress.add_task(
        "Fixing rotation vectors...", total=dataset.meta.total_episodes
    )
    current_episode_index = 0

    # Track previous rotation vectors (reset at episode boundaries)
    previous_cartesian_rotation = None
    previous_target_rotation = None

    for frame in dataset:
        # Check if we've moved to a new episode
        if frame["episode_index"] > current_episode_index:
            current_episode_index = frame["episode_index"]
            progress.update(progress_task, advance=1)
            new_dataset.save_episode()

            # Reset previous rotation vectors for new episode
            previous_cartesian_rotation = None
            previous_target_rotation = None

            # Check episode limit
            if LIMIT_EPISODES is not None and current_episode_index >= LIMIT_EPISODES:
                break

        # Create new frame with fixed rotation vectors
        new_frame = {}

        # Process each feature
        for key in dataset.features.keys():
            if key == "action":
                new_frame[key] = frame[key].cpu().numpy()
            elif key.startswith("observation.images"):
                image = einops.rearrange(frame[key], "c h w -> h w c")
                new_frame[key] = image
            elif key == "observation.state":
                # Fix rotation vectors in the concatenated state
                state = frame[key].clone().detach().cpu().numpy()

                # Fix cartesian rotation
                if cartesian_rotation_indices:
                    cartesian_rot = state[cartesian_rotation_indices]
                    fixed_cartesian_rot = _flip_rotation_vector_if_needed(
                        previous_cartesian_rotation, cartesian_rot
                    )
                    state[cartesian_rotation_indices] = fixed_cartesian_rot
                    previous_cartesian_rotation = fixed_cartesian_rot

                # Fix target rotation
                if target_rotation_indices:
                    target_rot = state[target_rotation_indices]
                    fixed_target_rot = _flip_rotation_vector_if_needed(
                        previous_target_rotation, target_rot
                    )
                    state[target_rotation_indices] = fixed_target_rot
                    previous_target_rotation = fixed_target_rot

                new_frame[key] = state
            elif key == "observation.state.cartesian":
                # Fix the cartesian feature directly
                cartesian = frame[key].clone().detach().cpu().numpy()
                cartesian_rot = cartesian[3:6]
                fixed_rot = _flip_rotation_vector_if_needed(
                    previous_cartesian_rotation, cartesian_rot
                )
                cartesian[3:6] = fixed_rot
                new_frame[key] = cartesian
            elif key == "observation.state.target":
                # Fix the target feature directly
                target = frame[key].clone().detach().cpu().numpy()
                target_rot = target[3:6]
                fixed_rot = _flip_rotation_vector_if_needed(previous_target_rotation, target_rot)
                target[3:6] = fixed_rot
                new_frame[key] = target
            elif key.startswith("observation.state."):
                # Handle scalar values (like gripper) and regular arrays
                if not frame[key].shape:
                    new_frame[key] = np.array([frame[key]], dtype=np.float32)
                else:
                    new_frame[key] = frame[key].cpu().numpy().squeeze()

        new_dataset.add_frame(new_frame, task=frame.get("task", ""))

    # Save final episode
    new_dataset.save_episode()
    progress.update(progress_task, advance=1)

print("\nRotation vector fix completed.")
print(f"Processed {current_episode_index + 1} episodes")
new_dataset.push_to_hub()
print(f"Dataset pushed to hub: {new_repo_id}")
