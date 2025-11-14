import cv2
import numpy as np
import pickle
import os

from src.parser.live_video_parser import VideoParser
from src.util.visualizer import visualize_pose
from src.model.media_pipe_pose import MediaPipePose
from src.util.visualizer import visualize_preds_vs_ground_truths


# load data keypoints
def load_keypoints(keypoint_path):
    with open(keypoint_path, "rb") as f:
        data = pickle.load(f)

    # AIST files typically have:
    # data["keypoints2d"]: shape (F, K, 3) or (F, K, 2)
    if "keypoints2d" in data:
        kpts = data["keypoints2d"][..., :2]  # drop confidence if exists
    elif "positions_2d" in data:
        kpts = data["positions_2d"][..., :2]
    else:
        raise ValueError("Keypoint structure unknown. Keys:", data.keys())

    return kpts


# run optical flow algorithm on keypoints
def lucas_kanade_on_keypoints(video_path, keypoint_path):
    # load ground truth keypoints
    gt_keypoints = load_keypoints(keypoint_path)
    num_frames, num_joints, _ = gt_keypoints.shape

    # load video
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        print("Error opening video")
        return

    # convert first frame to gray
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # convert first keypoints → float32 Nx1x2
    p0 = gt_keypoints[0].astype(np.float32).reshape(-1, 1, 2)

    # Lucas–Kanade params
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # visualization
    mask = np.zeros_like(old_frame)
    color = np.random.randint(0, 255, (num_joints, 3)).astype(np.uint8)

    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= num_frames:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # run Lucas–Kanade optical flow from old keypoints to new frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw tracks
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)

            # line from previous to new location
            mask = cv2.line(mask, (a, b), (c, d), color[j].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 4, color[j].tolist(), -1)

        img = cv2.add(frame, mask)

        # draw ground truth keypoints for reference
        for j in range(num_joints):
            gx, gy = gt_keypoints[frame_idx, j].astype(int)
            cv2.circle(img, (gx, gy), 3, (0, 255, 0), -1)

        cv2.imshow("Tracked Keypoints (LK) + GT", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # update
        old_gray = frame_gray.copy()
        p0 = p1.copy()
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


# overlay MediaPipe on top of AIST keypoints
def overlay_mediapipe_vs_aist(video_path, aist_kpts_path, out_path):
    frame_keypoints = load_keypoints(aist_kpts_path)
    parser = VideoParser(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = 60
    w = int(parser.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(parser.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    model = MediaPipePose(static_image_mode=True, model_complexity=0)

    frame_idx = 0

    for keypts, pkt in zip(frame_keypoints, parser):
        frame = pkt["frame"]
        pred = model.detect_landmarks(frame)
        pred = model.convert_to_aist17(pred)

        # visualization
        frame = visualize_preds_vs_ground_truths(
            frame, pred, keypts
        )

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    writer.release()
    print(f"Done. Saved: {out_path}")



def main():
    video_path = "data/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"
    keypoint_path = "data/keypoints/gBR_sBM_cAll_d04_mBR0_ch01.pkl"

    # optical flow keypoint tracking
    lucas_kanade_on_keypoints(video_path, keypoint_path)

    # mediapipe overlay
    overlay_mediapipe_vs_aist(video_path, keypoint_path, out_path="mediapipe_gt_overlay.mp4")

if __name__ == "__main__":
    main()
