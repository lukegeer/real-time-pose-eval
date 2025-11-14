import cv2
import numpy as np
import pickle


# AIST++ Official Parameters
AIST_W = 1920
AIST_H = 1080


# Load AIST 2D Keypoints (already in correct pixel coords)
def load_keypoints(path, view=0):
    with open(path, "rb") as f:
        data = pickle.load(f)

    kpts = data["keypoints2d"]           # (9, F, 17, 3)
    kpts = kpts[view]                    # (F, 17, 3)
    return kpts[..., :2].astype(np.float32)   # drop confidence


# Lucas–Kanade Tracking on AIST 2D keypoints
def lucas_kanade_on_keypoints(video_path, keypoint_path, view=0):
    gt_keypoints = load_keypoints(keypoint_path, view)
    num_frames, num_joints, _ = gt_keypoints.shape

    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        print("Error reading video")
        return

    # Resize video to AIST resolution
    old_frame = cv2.resize(old_frame, (AIST_W, AIST_H))
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Initial keypoints
    p0 = gt_keypoints[0].reshape(-1, 1, 2)

    # LK parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Random colors per joint
    colors = np.random.randint(0, 255, (num_joints, 3)).astype(np.uint8)

    # Mask to draw optical flow paths
    mask = np.zeros_like(old_frame)

    # Regular resizable window
    cv2.namedWindow("LK AIST Keypoints", cv2.WINDOW_NORMAL)

    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= num_frames:
            break

        frame = cv2.resize(frame, (AIST_W, AIST_H))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run Lucas–Kanade optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw optical flow paths on persistent mask
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), colors[j].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 4, colors[j].tolist(), -1)

        # Draw ground-truth keypoints
        for j in range(num_joints):
            gx, gy = gt_keypoints[frame_idx, j].astype(int)
            cv2.circle(frame, (gx, gy), 3, (0, 255, 0), -1)

        # Overlay mask on frame
        img = cv2.add(frame, mask)

        cv2.imshow("LK AIST Keypoints", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # Update previous frame/keypoints
        old_gray = frame_gray.copy()
        p0 = p1.copy()
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# Main
def main():
    video_path = "data/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"
    keypoint_path = "data/keypoints/gBR_sBM_cAll_d04_mBR0_ch01.pkl"

    lucas_kanade_on_keypoints(video_path, keypoint_path, view=0)


if __name__ == "__main__":
    main()
