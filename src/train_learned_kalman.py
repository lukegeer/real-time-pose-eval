import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.pose_sequence_dataset import PoseSequenceDataset
from src.model.learned_kalman import LearnedKalmanDynamics, rollout


def split_files(root, val_ratio=0.1):
    files = sorted(glob.glob(os.path.join(root, '*.npy')))
    np.random.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio))
    return files[n_val:], files[:n_val]


def train(args):
    torch.manual_seed(42)
    train_files, val_files = split_files(args.data_root, val_ratio=0.1)
    train_ds = PoseSequenceDataset(train_files, history=args.history, rollout=args.rollout)
    val_ds = PoseSequenceDataset(val_files, history=args.history, rollout=args.rollout)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LearnedKalmanDynamics(num_joints=args.num_joints, hidden_size=args.hidden_size, history=args.history, dt=args.dt).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss(reduction='none')

    def run_epoch(loader, train_mode=True):
        total = 0.0
        count = 0
        model.train(train_mode)
        for batch in loader:
            states = batch['states'].to(device)  # [B, window, J, 4]
            vis = batch['vis'].to(device)        # [B, window, J]
            init = states[:, 0]                  # [B, J, 4]
            gt = states[:, 1:1 + args.rollout]   # [B, T, J, 4]
            mask = vis[:, 1:1 + args.rollout]

            preds = rollout(model, init_state=init, gt_states=gt, mask=mask)
            mse = loss_fn(preds, gt)  # [B, T, J, 6]
            valid = mask.unsqueeze(-1)  # [B, T, J, 1]
            denom = valid.sum()
            if denom == 0:
                continue
            # Weight occluded joints higher for stability (when vis ~0)
            occluded = (1 - valid)
            loss_vis = (mse * valid).sum() / denom
            loss_occ = (preds[..., :2] * occluded).pow(2).sum() / max(occluded.sum(), 1.0)  # penalize movement when occluded
            loss = loss_vis + args.occluded_weight * loss_occ

            if train_mode:
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            total += loss.item()
            count += 1
        return total / max(count, 1)

    best_val = float('inf')
    for epoch in range(args.epochs):
        train_loss = run_epoch(train_loader, train_mode=True)
        val_loss = run_epoch(val_loader, train_mode=False)
        print(f"Epoch {epoch}: train {train_loss:.4f}, val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out)
    print(f"Best val {best_val:.4f}, saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Folder of .npy pose sequences')
    parser.add_argument('--num_joints', type=int, default=17)
    parser.add_argument('--history', type=int, default=3)
    parser.add_argument('--rollout', type=int, default=10)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--dt', type=float, default=1/30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--occluded_weight', type=float, default=0.1, help='Extra weight on stability when occluded')
    parser.add_argument('--out', type=str, default='checkpoints/learned_kalman.pth')
    args = parser.parse_args()
    train(args)
