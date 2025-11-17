import os
import sys
import json
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from model.embedding_model import PoseEmbeddingNet
from util.evaluation import compute_joint_angles, evaluate_similarity_metrics, save_similarity_metrics_table

def compute_oks(kp1, kp2, sigma=0.5):
    xy1 = kp1[:, :2]
    xy2 = kp2[:, :2]
    conf1 = kp1[:, 2]
    conf2 = kp2[:, 2]
    squared_dist = np.sum((xy1 - xy2) ** 2, axis=1)
    oks = np.exp(-squared_dist / (2 * sigma ** 2))
    weights = np.minimum(conf1, conf2)
    oks_weighted = np.sum(weights * oks) / (np.sum(weights) + 1e-8)
    return oks_weighted

def main():
    # Load model
    model = PoseEmbeddingNet(embed_dims=128, hidden_dims=[512, 512], dropout=0.3)
    model.load_state_dict(torch.load('../checkpoints/best_model.pth', map_location='cpu'))
    model.eval()

    # Load validation pairs
    with open('../data/raw/splits/val_pairs.json', 'r') as f:
        val_pairs = json.load(f)
    keypoints_folder = '../data/processed/aist_plusplus_final/keypoints2d'

    angle_sims = []
    oks_sims = []
    embed_sims = []
    labels = []

    for pair in val_pairs:
        # Load keypoints
        with open(os.path.join(keypoints_folder, pair['file1']), 'rb') as f:
            data1 = pickle.load(f)
        with open(os.path.join(keypoints_folder, pair['file2']), 'rb') as f:
            data2 = pickle.load(f)
        kp1 = np.array(data1['keypoints2d'][0][pair['frame1']])
        kp2 = np.array(data2['keypoints2d'][0][pair['frame2']])

        # Joint angle similarity (cosine)
        angles1 = compute_joint_angles(kp1)
        angles2 = compute_joint_angles(kp2)
        angle_sim = np.dot(angles1, angles2) / (np.linalg.norm(angles1) * np.linalg.norm(angles2) + 1e-8)
        angle_sims.append(angle_sim)

        # OKS similarity
        oks_sim = compute_oks(kp1, kp2)
        oks_sims.append(oks_sim)

        # Embedding similarity
        with torch.no_grad():
            emb1 = model(torch.tensor(kp1, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            emb2 = model(torch.tensor(kp2, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            embed_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
        embed_sims.append(embed_sim)

        # Label
        labels.append(pair['match'])

    angle_sims = np.array(angle_sims)
    oks_sims = np.array(oks_sims)
    embed_sims = np.array(embed_sims)
    labels = np.array(labels)

    # Evaluate and plot
    results = evaluate_similarity_metrics(angle_sims, oks_sims, embed_sims, labels)
    save_similarity_metrics_table(results)
    # Print summary table
    print("\nSummary Table:")
    print("{:<12} {:<12} {:<12}".format("Metric", "Threshold", "Accuracy"))
    for metric, vals in results.items():
        print("{:<12} {:<12.3f} {:<12.3f}".format(metric.capitalize(), vals['threshold'], vals['accuracy']))

if __name__ == '__main__':
    main()
