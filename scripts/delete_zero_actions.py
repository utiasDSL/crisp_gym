import numpy as np

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)

def action_nonzero(frame):
    """Check if the action in the frame is zero."""
    action = frame["action"]
    if np.linalg.norm(action[:3]) <= 1e-3:  # Check if the first 6 elements of the action are close to zero
        print(f"Frame {frame['timestamp']} has zero action: {action}")
        return False
    else:
        return True
    
def clean_action(action):
    """Clean the action by setting the first 6 elements to zero if they are close to zero."""
    set_to_zero_dims = [3, 4]
    action[set_to_zero_dims] = 0.0
    return action
    

source_repo_id = "ralfroemer/pick_lego_block"
target_repo_id = source_repo_id + "_filtered"

# Copy all frames from the source dataset to the target dataset, but filter out frames with zero actions.
source_dataset = LeRobotDataset(repo_id=source_repo_id)

# If the target dataset already exists, take it 
dataset = LeRobotDataset.create(
    repo_id=target_repo_id,
    fps=source_dataset.meta.fps,
    robot_type=source_dataset.meta.robot_type,
    features=source_dataset.meta.features,
    use_videos=False,
)

# Iterate through each episode
for episode_idx in range(source_dataset.num_episodes):
    # Get episode data range
    ep_start = source_dataset.episode_data_index["from"][episode_idx]
    ep_end = source_dataset.episode_data_index["to"][episode_idx]
    
    # Track if we have any frames to save for this episode
    has_frames = False
    
    # Process each frame in the episode
    n_frames = 0
    gripper_previous = None
    for frame_idx in range(ep_start, ep_end):
        frame_data = source_dataset.hf_dataset[frame_idx]
        
        # Check if action is zero (adjust threshold as needed)
        action = frame_data["action"]
        
        # Skip frames with zero action (using small threshold for numerical precision)
        if gripper_previous is None:
            gripper_unchanged = True
        else:
            gripper_unchanged = abs(action[6] - gripper_previous) <= 5e-2
        gripper_previous = action[6]

        if np.linalg.norm(action[:3]) <= 1e-3 and gripper_unchanged:
            # print(f"Skipping frame {frame_data['timestamp']} in episode {episode_idx} due to zero action")
            continue

        # Clean the action if needed
        action = clean_action(action)
            
        n_frames += 1
        # Convert frame data to the format expected by add_frame
        frame_dict = {}
        for key, value in frame_data.items():
            if 'images' in key:
                # Convert from (3, 256, 256) to (256, 256, 3) and from between [0, 1] to [0, 255]
                if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[0] == 3:
                    frame_dict[key] = (value.transpose(1, 2, 0) * 255).astype(np.uint8)
                else:   # Torch tensors
                    frame_dict[key] = (value.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            elif key not in ["index", "episode_index", "frame_index", "timestamp", "task_index"]:
                frame_dict[key] = np.asarray(value)
        
        # Add the frame to the target dataset
        task = source_dataset.meta.tasks[int(frame_data['task_index'])]
        dataset.add_frame(frame_dict, task=task)
        has_frames = True
    
    # Only save the episode if it has frames
    if has_frames:
        dataset.save_episode()
        print(f"Reducing episode length from {ep_end - ep_start} to {n_frames} frames for episode {episode_idx}")
    else:
        # Clear the buffer if no frames were added
        dataset.clear_episode_buffer()
        print(f"Episode {episode_idx} had no non-zero action frames, skipping")

print(f"Filtered dataset saved as {target_repo_id}")

dataset.push_to_hub(repo_id=target_repo_id, private=True)