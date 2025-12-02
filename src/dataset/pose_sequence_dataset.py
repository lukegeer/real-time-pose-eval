import numpy as np
import torch
from torch.utils.data import Dataset


class PoseSequenceDataset(Dataset):
    """
    Loads npy dicts with {'keypoints': [T, J, 3], 'vis': [T, J]} in pixels.
    Returns a fixed window of length window = history + rollout + 1 so batches collate.
    """

    def __init__(self, file_list, history=3, rollout=10):
        self.files = file_list
        self.history = history
        self.rollout = rollout
        self.window = history + rollout + 1
        self.joint_dropout = 0.1  # probability to drop a joint per frame
        self.limb_dropout = 0.05  # probability to drop all joints in a limb per frame
        self.limbs = [
            (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12), (5, 11), (6, 12)
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        kp = data['keypoints'].astype(np.float32)  # [T, J, 3]
        vis = data.get('vis', (kp[..., 2] > 0.3).astype(np.float32))
        # Invalidate non-finite keypoints
        finite = np.isfinite(kp).all(axis=-1)
        vis = vis * finite.astype(np.float32)
        kp[~finite] = 0.0
        # Stronger vis threshold
        vis = vis * (kp[..., 2] > 0.5).astype(np.float32)

        T = kp.shape[0]
        if T < self.window:
            raise ValueError(f"Sequence too short: {T} < {self.window} in {self.files[idx]}")
        start = np.random.randint(0, T - self.window + 1)
        kp = kp[start:start + self.window]
        vis = vis[start:start + self.window]

        xy = kp[..., :2]
        conf = kp[..., 2]
        # Clamp coords inside observed range
        xy_max = np.maximum(1.0, np.nanmax(xy))
        xy = np.clip(xy, 0.0, xy_max)
        # Normalize to [0,1]
        xy_norm = xy / xy_max
        vx = np.zeros_like(xy_norm)
        vx[1:] = xy_norm[1:] - xy_norm[:-1]
        ax = np.zeros_like(xy_norm)
        ax[2:] = vx[2:] - vx[1:-1]
        states = np.concatenate([xy_norm, vx, ax], axis=-1)  # [window, J, 6]

        # Occlusion/dropout augmentation
        if np.random.rand() < self.joint_dropout:
            drop_mask = (np.random.rand(*vis.shape) < self.joint_dropout).astype(np.float32)
            vis = vis * (1 - drop_mask)
        if np.random.rand() < self.limb_dropout:
            for (a, b) in self.limbs:
                if np.random.rand() < self.limb_dropout:
                    vis[:, a] = 0.0
                    vis[:, b] = 0.0

        return {
            'states': torch.from_numpy(states),
            'vis': torch.from_numpy(vis.astype(np.float32)),
            'conf': torch.from_numpy(conf.astype(np.float32)),
            'scale': xy_max,
        }
