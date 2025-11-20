import cv2
import numpy as np
import pickle


# ============================================================
# AIST++ Official Parameters
# ============================================================
AIST_W = 1920
AIST_H = 1080


# ============================================================
# Load AIST 2D Keypoints (already in correct pixel coords)
# ============================================================
def load_keypoints(path, view=0):
    with open(path, "rb") as f:
        data = pickle.load(f)
    kpts = data["keypoints2d"]           # (9, F, 17, 3)
    kpts = kpts[view]                    # (F, 17, 3)
    return kpts[..., :2].astype(np.float32)   # drop confidence


# ============================================================
# Dense Flow Keypoint Tracking
# ============================================================
def dense_flow_keypoint_tracking(video_path, keypoint_path, max_motion=30.0):
    """
    Tracks keypoints using dense optical flow (Farneback) to avoid LK teleportation.
    """
    gt_keypoints = load_keypoints(keypoint_path)
    num_frames, num_joints, _ = gt_keypoints.shape

    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        print("Error reading video")
        return

    # Resize to AIST resolution
    old_frame = cv2.resize(old_frame, (AIST_W, AIST_H))
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Start from the first GT keypoints
    tracked_points = gt_keypoints[0].copy()

    # Random colors per joint
    colors = np.random.randint(0, 255, (num_joints, 3)).astype(np.uint8)

    # Mask for drawing trails
    mask = np.zeros_like(old_frame)

    # Resizable window
    cv2.namedWindow("Dense Flow Keypoints", cv2.WINDOW_NORMAL)

    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= num_frames:
            break

        frame = cv2.resize(frame, (AIST_W, AIST_H))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, frame_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        new_points = tracked_points.copy()

        # Move each keypoint by sampled flow
        for j in range(num_joints):
            x, y = tracked_points[j]
            x_int = int(np.clip(x, 0, AIST_W-1))
            y_int = int(np.clip(y, 0, AIST_H-1))
            dx, dy = flow[y_int, x_int]

            # Apply motion threshold
            if np.hypot(dx, dy) < max_motion:
                new_points[j] = [x + dx, y + dy]

        # Draw optical flow paths
        for j in range(num_joints):
            a, b = tracked_points[j].astype(int)
            c, d = new_points[j].astype(int)
            mask = cv2.line(mask, (a, b), (c, d), colors[j].tolist(), 2)
            frame = cv2.circle(frame, (c, d), 4, colors[j].tolist(), -1)

        # Draw GT keypoints for comparison (green)
        for j in range(num_joints):
            gx, gy = gt_keypoints[frame_idx, j].astype(int)
            cv2.circle(frame, (gx, gy), 3, (0, 255, 0), -1)

        # Overlay mask
        img = cv2.add(frame, mask)
        cv2.imshow("Dense Flow Keypoints", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Update for next frame
        old_gray = frame_gray.copy()
        tracked_points = new_points.copy()
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Main
# ============================================================
def main():
    video_path = "data/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"
    keypoint_path = "data/keypoints/gBR_sBM_cAll_d04_mBR0_ch01.pkl"

    dense_flow_keypoint_tracking(video_path, keypoint_path, max_motion=30.0)


if __name__ == "__main__":
    main()
