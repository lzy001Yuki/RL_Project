"""
Behavioral cloning (BC) pretraining for the PPO policy.

This script trains the policy to imitate expert (A*) actions using a simple
maximum-likelihood objective under the policy's squashed Normal distribution.

Usage:
  cd _instructor
  python bc_pretrain.py \
    --demos_path ../data/astar_demos_eval20_res2.npz \
    --save_path saved_data/bc_controller.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import Policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--demos_path", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--entropy_coef",
        type=float,
        default=1e-3,
        help="Entropy bonus during BC to avoid collapsing exploration (0 disables).",
    )
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


@dataclass
class DemoBatch:
    obs_sensor: torch.Tensor  # (B, 4)
    action: torch.Tensor  # (B, 2)


class DemoDataset(Dataset):
    def __init__(self, demos_path: str):
        data = np.load(demos_path)
        if "obs_sensor" not in data or "actions" not in data:
            raise ValueError(f"Invalid demos file: {demos_path} (expected obs_sensor/actions)")
        self.obs_sensor = data["obs_sensor"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        if self.obs_sensor.ndim != 2 or self.obs_sensor.shape[1] != 4:
            raise ValueError(f"obs_sensor must be (M, 4), got {self.obs_sensor.shape}")
        if self.actions.ndim != 2 or self.actions.shape[1] != 2:
            raise ValueError(f"actions must be (M, 2), got {self.actions.shape}")

    def __len__(self) -> int:
        return int(self.obs_sensor.shape[0])

    def __getitem__(self, idx: int) -> DemoBatch:
        obs = torch.from_numpy(self.obs_sensor[idx])
        act = torch.from_numpy(self.actions[idx])
        return DemoBatch(obs_sensor=obs, action=act)


def _collate_fn(batch: list[DemoBatch]) -> DemoBatch:
    obs = torch.stack([b.obs_sensor for b in batch], dim=0)
    act = torch.stack([b.action for b in batch], dim=0)
    return DemoBatch(obs_sensor=obs, action=act)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset = DemoDataset(args.demos_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=_collate_fn,
    )
    print(f"Loaded demos: {len(dataset)} samples from {args.demos_path}")

    policy = Policy(obs_dim=4, action_dim=2, action_limit=(2.0, 2.0)).to(device)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    action_scale = policy.action_scale.to(device)

    for epoch in range(args.epochs):
        total_nll = 0.0
        total_mse = 0.0
        n_batches = 0

        for batch in loader:
            obs = {"sensor": batch.obs_sensor.to(device)}
            expert_actions = batch.action.to(device)

            # Negative log-likelihood of expert actions under the policy distribution.
            _, logp, entropy, _ = policy.evaluate_actions(obs, actions=expert_actions)
            nll = -logp.mean()

            # Auxiliary: MSE between deterministic mean action and expert action (for monitoring).
            feat = policy.forward(obs)
            mean_unscaled = policy.actor_mean(feat)
            mean_scaled = torch.tanh(mean_unscaled) * action_scale
            mse = torch.mean((mean_scaled - expert_actions) ** 2)

            loss = nll - args.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            total_nll += float(nll.detach().cpu())
            total_mse += float(mse.detach().cpu())
            n_batches += 1

        print(
            f"[BC epoch {epoch:03d}] "
            f"nll={total_nll / max(n_batches, 1):.4f}  "
            f"mse={total_mse / max(n_batches, 1):.4f}  "
            f"batches={n_batches}"
        )

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(policy.state_dict(), args.save_path)
    print(f"Saved BC-pretrained weights to {args.save_path}")


if __name__ == "__main__":
    main()
