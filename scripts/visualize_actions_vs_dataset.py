import argparse  # noqa: D100
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class

#Anything that the policy should not read 
SUPERVISION_PREFIXES = ("action","target")  

def to_device_batch(batch: dict, device: torch.device, non_blocking: bool = True) -> dict:  # noqa: D103
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out

def _make_inference_batch(batch: dict) -> dict:
    cleaned = {}
    for k, v in batch.items():
        # keep only observation/metadata; drop action-like keys
        if any(k == p or k.startswith(p) for p in SUPERVISION_PREFIXES):
            continue
        cleaned[k] = v
    return cleaned

parser = argparse.ArgumentParser(description="Visualize policy vs. dataset actions")
parser.add_argument(
    "--checkpoint",
    type=Path,
    required=True,
    help="Path to the policy checkpoint directory (e.g., outputs/train/.../pretrained_model).",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Path to the dataset directory OR a Hub repo id (e.g., lerobot/aloha_static_coffee).",
)
parser.add_argument(
    "--episode",
    type=int,
    default=0,
    help="Episode index to compare (default: 0).",
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Optional path for saving the matplotlib figure. "
            "If omitted, saved as actions_episode<EP>.png in cwd",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=8,
    help="Dataloader workers (default: 8).",
)
args = parser.parse_args()

# Find out the device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare the policy
train_config = TrainPipelineConfig.from_pretrained(args.checkpoint)
policy_cls = get_policy_class(train_config.policy.type)
policy = policy_cls.from_pretrained(args.checkpoint)
policy.to(device)
policy.eval()

# Load the dataset 
dataset = LeRobotDataset(repo_id=args.dataset, root=args.dataset)

# Build the data loader 
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=args.num_workers,
    batch_size=1,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
    drop_last=False,
)

ep = args.episode
targets: List[torch.Tensor] = []
preds: List[torch.Tensor] = []
times: List[float] = []
found_any = False
action_dim: Optional[int] = None


last_batch: Optional[Dict[str, Any]] = None
for batch in dataloader:

    # Try to find the correct batch
    b_ep = batch.get("episode_index")
    if b_ep is None:
        raise KeyError("Expected key 'episode_index' in batch.")
    b_ep = int(b_ep.view(-1)[0].item())
 
    if b_ep < ep:
        continue
    if b_ep > ep:
        break

    found_any = True
    last_batch = batch

    batch = to_device_batch(batch, device, non_blocking=True)

    tgt = batch["action"].detach().float().view(-1)

    if action_dim is None:
        action_dim = tgt.numel()

    # Predicted action from policy
    cleaned_batch=_make_inference_batch(batch)
    pred = policy.select_action(cleaned_batch)
    pred = pred.detach().float().view(-1)

    # Collect
    targets.append(tgt.cpu())
    preds.append(pred.cpu())

    # Time (fallback to step index if timestamp is absent)
    if "timestamp" in batch:
        t = float(batch["timestamp"].view(-1)[0].detach().cpu().item())
    else:
        # reconstruct pseudo-time from frame_index if available; else just enumerate
        if "frame_index" in batch:
            t = float(batch["frame_index"].view(-1)[0].detach().cpu().item())
        else:
            t = float(len(times))
    times.append(t)

if not found_any:
    raise ValueError(f"No frames found for episode_index={ep}. "
                        "Check that the dataset contains this episode.")

# ----- Stack to T x D -----
targets_t = torch.stack(targets, dim=0)  # [T, D]
preds_t = torch.stack(preds, dim=0)      # [T, D]
times_t = torch.tensor(times)            # [T]

T, D = targets_t.shape

# ----- Plot -----
fig, axes = plt.subplots(D, 1, figsize=(9, 2.3 * D), sharex=True)
if D == 1:
    axes = [axes]

for d in range(D):
    ax = axes[d]
    ax.plot(times_t.numpy(), targets_t[:, d].numpy(), label="Target")
    ax.plot(times_t.numpy(), preds_t[:, d].numpy(), label="Pred")
    ax.set_ylabel(f"dim {d}")
    ax.grid(True, linestyle="--", alpha=0.3)
    if d == 0:
        ax.set_title(f"Episode {ep}: action targets vs. predictions")

xlabel = "time (s)" if (last_batch and "timestamp" in last_batch) else "step"
axes[-1].set_xlabel(xlabel)

# single legend outside if many dims
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")

plt.tight_layout(rect=[0, 0, 0.98, 0.98])

output_path = args.output or Path(f"./actions_episode{args.episode}.png")
fig.savefig(output_path, dpi=150)
plt.close(fig)
print(f"Saved continuous plot to {output_path}")

