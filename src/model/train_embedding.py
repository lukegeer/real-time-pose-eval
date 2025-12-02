from embedding_model import PoseEmbeddingNet, ContrastiveLoss
import json
import pickle
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class TripletPoseDataset(Dataset):
    def __init__(self, pairs_json, keypoints_folder, augment=False):
        with open(pairs_json, 'r') as f:
            self.pairs = json.load(f)

        self.keypoints_folder = keypoints_folder

        self.augment = augment
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]

        keypoints1 = self.load_keypoints(pair, 'file1', 'frame1')
        keypoints2 = self.load_keypoints(pair, 'file2', 'frame2')
        label = pair['match']

        if self.augment and np.random.rand() < 0.5:
            keypoints1 = self.augment_keypoints(keypoints1)
            keypoints2 = self.augment_keypoints(keypoints2)

        return {
            'keypoints1': torch.tensor(keypoints1, dtype=torch.float32),
            'keypoints2': torch.tensor(keypoints2, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }
    
    def load_keypoints(self, pair, file, frame):
        file_path = pair[file]
        file_path = os.path.join(self.keypoints_folder, file_path)

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        frame_idx = pair[frame]

        frame_keypoints = data['keypoints2d'][0][frame_idx]

        return frame_keypoints
    
    def augment_keypoints(self, keypoints):
        kp = keypoints.copy()
        xy = kp[:, :2]
        
        valid_mask = kp[:, 2] > 0
        if valid_mask.sum() == 0:
            return kp
        
        valid_xy = xy[valid_mask]
        bbox_min = valid_xy.min(axis=0)
        bbox_max = valid_xy.max(axis=0)
        bbox_size = bbox_max - bbox_min
        center = (bbox_min + bbox_max) / 2
        
        if np.random.rand() < 0.5:
            kp[:, 0] = center[0] + (center[0] - kp[:, 0])
            left_right_pairs = [(5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
            for left_idx, right_idx in left_right_pairs:
                kp[[left_idx, right_idx]] = kp[[right_idx, left_idx]]
        
        scale = np.random.uniform(0.9, 1.1)
        kp[:, :2] = center + (xy - center) * scale
        
        angle = np.random.uniform(-15, 15) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        kp[:, :2] = center + (kp[:, :2] - center) @ rotation_matrix.T
        
        translation = np.random.uniform(-0.05, 0.05) * bbox_size
        kp[:, :2] += translation
        
        noise_scale = 0.01 * bbox_size
        noise = np.random.normal(0, 1, size=kp[:, :2].shape) * noise_scale
        kp[:, :2] += noise
        
        conf_mask = np.random.rand(len(kp)) > 0.1
        kp[:, 2] *= conf_mask
        
        return kp
    

def train_epoch(model, criterion, train_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc='Training'):
        kp1 = batch['keypoints1'].to(device)
        kp2 = batch['keypoints2'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        embed1 = model(kp1)
        embed2 = model(kp2)
        
        loss = criterion(embed1, embed2, kp1, kp2, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, criterion, val_loader, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            kp1 = batch['keypoints1'].to(device)
            kp2 = batch['keypoints2'].to(device)
            labels = batch['label'].to(device)
            
            embed1 = model(kp1)
            embed2 = model(kp2)
            
            loss = criterion(embed1, embed2, kp1, kp2, labels)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    keypoints_folder = '../../data/processed/aist_plusplus_final/keypoints2d'
    train_pairs = '../../data/raw/splits/train_pairs.json'
    val_pairs = '../../data/raw/splits/val_pairs.json'
    
    train_dataset = TripletPoseDataset(train_pairs, keypoints_folder, augment=False)
    val_dataset = TripletPoseDataset(val_pairs, keypoints_folder, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    
    model = PoseEmbeddingNet(embed_dims=128, hidden_dims=[512, 512], dropout=0.3).to(device)
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters.")
    criterion = ContrastiveLoss(margin=0.5, oks_weight=0.0, sigma=0.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=8,
    )
    
    best_val_loss = float('inf')
    for epoch in range(100):
        train_loss = train_epoch(model, criterion, train_loader, optimizer, device)
        val_loss = validate(model, criterion, val_loader, device)

        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../../checkpoints/best_model.pth')
            print("Saved best model")


if __name__ == '__main__':
    main()







