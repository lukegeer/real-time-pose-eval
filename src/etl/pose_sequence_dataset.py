import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PoseSequenceDataset(Dataset):
    """
    Expects npy files with dicts: {'keypoints': [T, J, 3], 'vis': [T, J]} optionally 'flow': [T, H, W, 2]
    Keypoints assumed in pixels; vis in [0,1].
    """

    def __init__(self, file_list, history=3, future=1):
        self.files = file_list
        self.history = history
        self.future = future

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        kp = data['keypoints'].astype(np.float32)  # [T, J, 3]
        vis = data.get('vis', (kp[..., 2] > 0.3).astype(np.float32))
        T = kp.shape[0]
        # Build states [T, J, 4]: x,y,vx,vy (vel finite diff)
        xy = kp[..., :2]
        vx = np.zeros_like(xy)
        vx[1:] = xy[1:] - xy[:-1]
        states = np.concatenate([xy, vx], axis=-1)
        return {
            'states': torch.from_numpy(states),
            'vis': torch.from_numpy(vis.astype(np.float32)),
        }
