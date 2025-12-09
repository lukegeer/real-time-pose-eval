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


def pairwise_distance_error(kp_a, kp_b, conf_threshold=0.1):
    """
    Sum of squared differences (normalized) between all pairwise joint distances for two poses.
    Uses only joints where both confidences >= conf_threshold; returns None if <2 joints valid.
    """
    conf_a = kp_a[:, 2]
    conf_b = kp_b[:, 2]
    valid = (conf_a >= conf_threshold) & (conf_b >= conf_threshold)
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return None
    xy_a = kp_a[idx, :2]
    xy_b = kp_b[idx, :2]
    # Normalize coordinates to reduce scale effects (use bbox diagonal)
    def normalize_xy(xy):
        min_xy = np.nanmin(xy, axis=0)
        max_xy = np.nanmax(xy, axis=0)
        span = max(np.linalg.norm(max_xy - min_xy), 1e-6)
        return (xy - min_xy) / span
    xy_a = normalize_xy(xy_a)
    xy_b = normalize_xy(xy_b)
    diff_a = xy_a[:, None, :] - xy_a[None, :, :]
    diff_b = xy_b[:, None, :] - xy_b[None, :, :]
    dist_a = np.linalg.norm(diff_a, axis=-1)
    dist_b = np.linalg.norm(diff_b, axis=-1)
    iu = np.triu_indices_from(dist_a, k=1)
    da = dist_a[iu]
    db = dist_b[iu]
    # Normalize pairwise distances themselves to unit mean to reduce global scale effects
    da = da / (np.mean(da) + 1e-6)
    db = db / (np.mean(db) + 1e-6)
    delta = da - db
    return float(np.sum(delta * delta))



