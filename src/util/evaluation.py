import torch
import numpy as np
import pickle
import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.embedding_model import PoseEmbeddingNet
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc

def compute_joint_angles(keypoints):
    xy = keypoints[:, :2]
    
    angles = []
    
    # Left elbow (shoulder -> elbow -> wrist)
    angles.append(angle_between_points(xy[5], xy[7], xy[9]))
    # Right elbow
    angles.append(angle_between_points(xy[6], xy[8], xy[10]))
    # Left knee
    angles.append(angle_between_points(xy[11], xy[13], xy[15]))
    # Right knee
    angles.append(angle_between_points(xy[12], xy[14], xy[16]))
    # Left shoulder
    angles.append(angle_between_points(xy[11], xy[5], xy[7]))
    # Right shoulder
    angles.append(angle_between_points(xy[12], xy[6], xy[8]))
    # Left hip
    angles.append(angle_between_points(xy[5], xy[11], xy[13]))
    # Right hip
    angles.append(angle_between_points(xy[6], xy[12], xy[14]))
    
    return np.array(angles)

def angle_between_points(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    
    if v1_norm < 1e-6 or v2_norm < 1e-6:
        return 0.0
    
    v1 = v1 / v1_norm
    v2 = v2 / v2_norm
    
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return angle

def find_best_threshold(scores, labels):
    thresholds = np.linspace(np.min(scores), np.max(scores), 100)
    best_acc = 0
    best_thr = 0
    best_pos_acc = 0
    best_neg_acc = 0
    for thr in thresholds:
        preds = (scores > thr).astype(int)
        acc = accuracy_score(labels, preds)
        pos_acc = ((preds == 1) & (labels == 1)).sum() / (labels == 1).sum()
        neg_acc = ((preds == 0) & (labels == 0)).sum() / (labels == 0).sum()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
            best_pos_acc = pos_acc
            best_neg_acc = neg_acc
    return best_thr, best_acc, best_pos_acc, best_neg_acc

def evaluate_similarity_metrics(angle_sims, oks_sims, embed_sims, labels):
    # Find best thresholds and accuracies
    thr_angle, acc_angle, pos_acc_angle, neg_acc_angle = find_best_threshold(angle_sims, labels)
    thr_oks, acc_oks, pos_acc_oks, neg_acc_oks = find_best_threshold(oks_sims, labels)
    thr_embed, acc_embed, pos_acc_embed, neg_acc_embed = find_best_threshold(embed_sims, labels)

    # Bar plot
    plt.figure(figsize=(6,4))
    plt.bar(['Joint Angle', 'OKS', 'Embedding'], [acc_angle, acc_oks, acc_embed], color=['#4e79a7','#f28e2b','#59a14f'])
    plt.ylabel('Best Accuracy')
    plt.title('Best Accuracy for Each Similarity Metric')
    plt.ylim(0, 1)
    plt.savefig("barplot_similarity_metrics.png", bbox_inches='tight', dpi=200)
    plt.show()

    # ROC curves
    plt.figure(figsize=(6,6))
    for sims, name, color in zip([angle_sims, oks_sims, embed_sims], ['Joint Angle', 'OKS', 'Embedding'], ['#4e79a7','#f28e2b','#59a14f']):
        fpr, tpr, _ = roc_curve(labels, sims)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.2f})', color=color)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Similarity Metrics')
    plt.legend()
    plt.savefig("roc_similarity_metrics.png", bbox_inches='tight', dpi=200)
    plt.show()

    print(f"Best thresholds:")
    print(f"  Joint Angle: {thr_angle:.3f} (acc={acc_angle:.3f}, pos_acc={pos_acc_angle:.3f}, neg_acc={neg_acc_angle:.3f})")
    print(f"  OKS:         {thr_oks:.3f} (acc={acc_oks:.3f}, pos_acc={pos_acc_oks:.3f}, neg_acc={neg_acc_oks:.3f})")
    print(f"  Embedding:   {thr_embed:.3f} (acc={acc_embed:.3f}, pos_acc={pos_acc_embed:.3f}, neg_acc={neg_acc_embed:.3f})")

    return {
        'angle': {'threshold': thr_angle, 'accuracy': acc_angle, 'pos_acc': pos_acc_angle, 'neg_acc': neg_acc_angle},
        'oks': {'threshold': thr_oks, 'accuracy': acc_oks, 'pos_acc': pos_acc_oks, 'neg_acc': neg_acc_oks},
        'embedding': {'threshold': thr_embed, 'accuracy': acc_embed, 'pos_acc': pos_acc_embed, 'neg_acc': neg_acc_embed},
    }


def save_similarity_metrics_table(results, csv_path='similarity_metrics_summary.csv', md_path='similarity_metrics_summary.md'):
    # Save as CSV
    with open(csv_path, 'w') as f:
        f.write("Metric,Threshold,Accuracy,Positive_Acc,Negative_Acc\n")
        for metric, vals in results.items():
            f.write(f"{metric.capitalize()},{vals['threshold']:.3f},{vals['accuracy']:.3f},{vals['pos_acc']:.3f},{vals['neg_acc']:.3f}\n")

    # Save as Markdown
    with open(md_path, 'w') as f:
        f.write("| Metric     | Threshold | Accuracy  | Pos Acc   | Neg Acc   |\n")
        f.write("|------------|-----------|-----------|-----------|-----------|\n")
        for metric, vals in results.items():
            f.write(f"| {metric.capitalize():<10} | {vals['threshold']:.3f}    | {vals['accuracy']:.3f}    | {vals['pos_acc']:.3f}    | {vals['neg_acc']:.3f}    |\n")

