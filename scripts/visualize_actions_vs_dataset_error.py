import argparse  # noqa: D100
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class

# Anything that the policy should not read
SUPERVISION_PREFIXES = ("action", "target")


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


def compute_rmse_series(targets_t: torch.Tensor, preds_t: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute RMSE metrics.
      - per_step_rmse[t] = sqrt(mean_d (pred[t,d]-target[t,d])^2)
      - cumulative_rmse[t] = sqrt(mean_{<=t,d} (pred-target)^2)
      - overall_rmse = cumulative_rmse[-1]
    All tensors are CPU tensors of shape [T] (except overall which is scalar).
    """
    assert targets_t.shape == preds_t.shape, "targets and preds must have same shape"
    diff = preds_t - targets_t  # [T, D]
    se_per_step = (diff ** 2).mean(dim=1)      # [T] mean over dims
    per_step_rmse = torch.sqrt(se_per_step)    # [T]

    # cumulative mean of squared error, then sqrt (running RMSE)
    cumsum_se = torch.cumsum(se_per_step, dim=0)               # [T]
    steps = torch.arange(1, se_per_step.numel() + 1, dtype=se_per_step.dtype)
    cumulative_rmse = torch.sqrt(cumsum_se / steps)            # [T]

    overall_rmse = cumulative_rmse[-1].item()

    return {
        "per_step_rmse": per_step_rmse,        # [T]
        "cumulative_rmse": cumulative_rmse,    # [T]
        "overall_rmse": overall_rmse,          # float
    }


def main():
    parser = argparse.ArgumentParser(
        description="Visualize policy vs. dataset actions with replan markers and RMSE plot"
    )
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
        help=(
            "Optional path for saving the matplotlib figure. "
            "If omitted, saved as actions_episode<EP>.png in cwd"
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Dataloader workers (default: 8).",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help=(
            "How many action steps the policy outputs per observation (e.g., 15). "
            "If omitted, falls back to checkpoint's n_action_steps (if available), else 1."
        ),
    )
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Policy
    train_config = TrainPipelineConfig.from_pretrained(args.checkpoint)
    policy_cls = get_policy_class(train_config.policy.type)
    policy_config = PreTrainedConfig.from_pretrained(args.checkpoint)

    # Determine action horizon
    if args.action_horizon is not None:
        action_horizon = int(args.action_horizon)
    else:
        action_horizon = int(getattr(policy_config, "n_action_steps", 1) or 1)

    policy = policy_cls.from_pretrained(args.checkpoint, config=policy_config)
    policy.to(device)
    policy.eval()

    # Dataset & loader
    dataset = LeRobotDataset(repo_id=args.dataset, root=args.dataset)
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
        # find the correct episode
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

        # Predicted action
        cleaned_batch = _make_inference_batch(batch)
        pred = policy.select_action(cleaned_batch)
        pred = pred.detach().float().view(-1)

        # Collect
        targets.append(tgt.cpu())
        preds.append(pred.cpu())

        # Time
        if "timestamp" in batch:
            t = float(batch["timestamp"].view(-1)[0].detach().cpu().item())
        else:
            if "frame_index" in batch:
                t = float(batch["frame_index"].view(-1)[0].detach().cpu().item())
            else:
                t = float(len(times))
        times.append(t)

    if not found_any:
        raise ValueError(
            f"No frames found for episode_index={ep}. Check that the dataset contains this episode."
        )

    # ----- Stack to T x D -----
    targets_t = torch.stack(targets, dim=0)  # [T, D]
    preds_t = torch.stack(preds, dim=0)      # [T, D]
    times_t = torch.tensor(times)            # [T]

    T, D = targets_t.shape

    # ----- RMSE series -----
    rmse = compute_rmse_series(targets_t, preds_t)
    per_step_rmse = rmse["per_step_rmse"]          # [T]
    cumulative_rmse = rmse["cumulative_rmse"]      # [T]
    overall_rmse = rmse["overall_rmse"]            # float

    # Print a concise RMSE summary
    print("\n=== RMSE Summary ===")
    print(f"Episode: {ep}")
    print(f"Overall RMSE: {overall_rmse:.6f}")
    print("====================\n")

    # Replan boundaries
    replan_idx = list(range(action_horizon, T, action_horizon)) if action_horizon > 0 else []

    # ----- Plot -----
    # We make D action subplots + 1 RMSE subplot at the bottom
    fig, axes = plt.subplots(D + 1, 1, figsize=(9, 2.3 * D + 3.0), sharex=True)
    axes = list(axes)  # ensure list

    # Action traces
    for d in range(D):
        ax = axes[d]
        ax.plot(times_t.numpy(), targets_t[:, d].numpy(), label="Target")
        ax.plot(times_t.numpy(), preds_t[:, d].numpy(), label="Pred")

        # Dots at replan boundaries (on top of the prediction curve)
        if replan_idx:
            ax.scatter(
                times_t[replan_idx].numpy(),
                preds_t[replan_idx, d].numpy(),
                s=30,
                marker="o",
                facecolors="none",
                edgecolors="k",
                linewidths=1.0,
                label="New prediction start" if d == 0 else None,
                zorder=3,
            )

        # Put a compact legend only on the first action subplot (to reduce clutter)
        if d == 0:
            ax.legend(loc="upper left")
            ax.set_title(
                f"Episode {ep}: action targets vs. predictions (horizon={action_horizon})"
            )

        ax.set_ylabel(f"dim {d}")
        ax.grid(True, linestyle="--", alpha=0.3)

    # RMSE subplot (instantaneous and cumulative)
    loss_ax = axes[-1]
    loss_ax.plot(times_t.numpy(), per_step_rmse.numpy(), label="Per-step RMSE")
    loss_ax.plot(times_t.numpy(), cumulative_rmse.numpy(), label="Cumulative RMSE")

    if replan_idx:
        loss_ax.scatter(
            times_t[replan_idx].numpy(),
            per_step_rmse[replan_idx].numpy(),
            s=25,
            marker="x",
            label="New prediction start",
            zorder=3,
        )

    # Put overall RMSE as text inside the loss subplot (so legend won't cover it)
    text_str = f"Overall RMSE = {overall_rmse:.4f}"
    loss_ax.text(
        0.01, 0.95, text_str,
        transform=loss_ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, linewidth=0.5)
    )

    loss_ax.set_ylabel("RMSE")
    loss_ax.legend(loc="upper right")
    loss_ax.grid(True, linestyle="--", alpha=0.3)

    xlabel = "time (s)" if (last_batch and "timestamp" in last_batch) else "step"
    axes[-1].set_xlabel(xlabel)

    plt.tight_layout(rect=[0, 0, 0.98, 0.98])

    output_path = args.output or Path(f"./actions_episode{args.episode}.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot (with instantaneous & cumulative RMSE) to {output_path}")


if __name__ == "__main__":
    main()
