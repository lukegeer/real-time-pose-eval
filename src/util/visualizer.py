import cv2
import numpy as np

LIMBS = [
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
    (5,6),(11,12),(5,11),(6,12)
]

def visualize_pose(frame, keypoints, color=(0,255,0), position=(10, 30), show_sum_keypoints=False, confidence_threshold=0.3, show_fps=False, fps=60):
    sum_keypoints = 0
    for (x, y, conf) in keypoints:
        if conf > confidence_threshold and not (np.isnan(x) or np.isnan(y)):
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            sum_keypoints += 1
    if show_sum_keypoints:
        cv2.putText(frame, f'Keypoints: {sum_keypoints}', position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if show_fps:
        cv2.putText(frame, f'FPS: {fps:.1f}', (position[0], position[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for (a,b) in LIMBS:
        if keypoints[a,2]>0.05 and keypoints[b,2]>0.05 and not (np.isnan(keypoints[a,0]) or np.isnan(keypoints[b,0]) or np.isnan(keypoints[a,1]) or np.isnan(keypoints[b,1])):
            pt1 = (int(keypoints[a,0]), int(keypoints[a,1]))
            pt2 = (int(keypoints[b,0]), int(keypoints[b,1]))
            cv2.line(frame, pt1, pt2, color, 2)
    return frame

def visualize_multiple(frame, pred_keypoints, gt_keypoints, pred_color=(255,0,0), gt_color=(0,255,0), fps=60):
    frame = visualize_pose(frame, pred_keypoints, pred_color, show_sum_keypoints=True, show_fps=True, fps=fps)
    frame = visualize_pose(frame, gt_keypoints, gt_color, position=(350, 30), show_sum_keypoints=True)
    return frame

