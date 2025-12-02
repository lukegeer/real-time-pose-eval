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
        if not (np.isscalar(x) and np.isscalar(y) and np.isscalar(conf)):
            continue
        if conf > confidence_threshold and np.isfinite(x) and np.isfinite(y):
            try:
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                sum_keypoints += 1
            except Exception:
                pass
    if show_sum_keypoints:
        cv2.putText(frame, f'Keypoints: {sum_keypoints}', position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if show_fps:
        cv2.putText(frame, f'FPS: {fps:.1f}', (position[0], position[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for (a,b) in LIMBS:
        xa, ya, ca = keypoints[a,0], keypoints[a,1], keypoints[a,2]
        xb, yb, cb = keypoints[b,0], keypoints[b,1], keypoints[b,2]
        if not (np.isscalar(xa) and np.isscalar(ya) and np.isscalar(xb) and np.isscalar(yb)):
            continue
        if ca>0.05 and cb>0.05 and np.isfinite(xa) and np.isfinite(ya) and np.isfinite(xb) and np.isfinite(yb):
            try:
                pt1 = (int(xa), int(ya))
                pt2 = (int(xb), int(yb))
                cv2.line(frame, pt1, pt2, color, 2)
            except Exception:
                pass
    return frame

def visualize_multiple(frame, pred_keypoints, gt_keypoints, pred_color=(255,0,0), gt_color=(0,255,0), fps=60):
    frame = visualize_pose(frame, pred_keypoints, pred_color, show_sum_keypoints=True, show_fps=True, fps=fps)
    frame = visualize_pose(frame, gt_keypoints, gt_color, position=(350, 30), show_sum_keypoints=True)
    return frame
