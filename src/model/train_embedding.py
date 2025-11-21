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
    def __init__(self, pairs_json, keypoints_folder):
        with open(pairs_json, 'r') as f:
            self.pairs = json.load(f)

        self.keypoints_folder = keypoints_folder
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]

        keypoints1 = self.load_keypoints(pair, 'file1', 'frame1')
        keypoints2 = self.load_keypoints(pair, 'file2', 'frame2')
        label = pair['match']

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
    
    train_dataset = TripletPoseDataset(train_pairs, keypoints_folder)
    val_dataset = TripletPoseDataset(val_pairs, keypoints_folder)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    model = PoseEmbeddingNet(embed_dims=64).to(device)
    criterion = ContrastiveLoss(margin=0.5, oks_weight=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    for epoch in range(50):
        train_loss = train_epoch(model, criterion, train_loader, optimizer, device)
        val_loss = validate(model, criterion, val_loader, device)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../../checkpoints/best_model.pth')
            print("Saved best model")


if __name__ == '__main__':
    main()







