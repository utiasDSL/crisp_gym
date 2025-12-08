"""Helper script to convert a dataset from relative actions to absolute ones."""

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.envs.utils import einops
from rich import print
from rich.progress import Progress

RELATIVE_ACTIONS_DATASET = "LSY-lab/stack_cake_v2"
dataset = LeRobotDataset(repo_id=RELATIVE_ACTIONS_DATASET)

print(dataset.meta.total_episodes)

# %%
# for frame in dataset:
#     image = einops.rearrange(frame["observation.images.wrist"], "c h w -> h w c")
#     break

# %%

features = {}

for key, value in dataset.features.items():
    if key.startswith("observation."):
        features[key] = value
    if key == "action":
        features[key] = value

if "observation.state.target" not in features:
    raise ValueError("Dataset does not contain target information.")
else:
    print("Target information found in the dataset.")

print("Features for the new dataset:")
for key in features.keys():
    print(f"- {key}: {features[key]['names']}")

# %%
new_dataset = LeRobotDataset.create(
    repo_id=f"{RELATIVE_ACTIONS_DATASET}_absolute_actions",
    features=features,
    fps=dataset.fps,
)

# %%

with Progress() as progress:
    progress_task = progress.add_task("Converting dataset...", total=dataset.meta.total_episodes)
    current_episode_index = 0
    for frame in dataset:
        new_frame = {}
        for key in features.keys():
            if key == "action":
                # Convert relative actions to absolute actions
                action = frame["action"]
                gripper_action = action[6]

                target_position = frame["observation.state.target"][:3]
                target_orientation = frame["observation.state.target"][3:6]

                absolute_action = np.array(
                    [
                        target_position[0],
                        target_position[1],
                        target_position[2],
                        target_orientation[0],
                        target_orientation[1],
                        target_orientation[2],
                        gripper_action,
                    ],
                    dtype=np.float32,
                )
                new_frame[key] = absolute_action
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

        new_dataset.add_frame(new_frame, task="Pick the lego block")

    new_dataset.save_episode()
    progress.update(progress_task, advance=1)

# %%
print("Conversion completed.")
new_dataset.push_to_hub()
