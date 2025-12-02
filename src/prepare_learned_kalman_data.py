import argparse
import glob
import os
import pickle

import cv2
import numpy as np


def load_keypoints(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # keypoints2d is typically [num_cams, T, J, 3]; use camera 0 (cAll)
    kp = np.array(data['keypoints2d'][0], dtype=np.float32)
    return kp


def process_pair(video_path, kp_path, out_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    kp = load_keypoints(kp_path)  # [T, J, 3] normalized
    xy = kp[..., :2].copy()
    conf = kp[..., 2]
    xy[..., 0] *= w
    xy[..., 1] *= h
    keypoints_px = np.concatenate([xy, conf[..., None]], axis=-1)
    vis = (conf > 0.3).astype(np.float32)

    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{basename}.npy")
    os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, {'keypoints': keypoints_px, 'vis': vis})
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_glob', type=str, default='data/raw/**/*.mp4', help='Glob for videos')
    parser.add_argument('--kp_root', type=str, default='data/processed/aist_plusplus_final/keypoints2d', help='Root for keypoint pickles')
    parser.add_argument('--out_dir', type=str, default='data/processed/learned_kalman', help='Output dir for npy sequences')
    args = parser.parse_args()

    videos = glob.glob(args.videos_glob, recursive=True)
    print(f"Found {len(videos)} videos")

    converted = []
    for vid in videos:
        base = os.path.splitext(os.path.basename(vid))[0]
        # Replace camera token with cAll to match GT filenames
        kp_base = base.replace('_c01_', '_cAll_').replace('_c02_', '_cAll_').replace('_c03_', '_cAll_')
        kp_path = os.path.join(args.kp_root, f"{kp_base}.pkl")
        if not os.path.exists(kp_path):
            print(f"Skip {vid}: no keypoints {kp_path}")
            continue
        try:
            out_path = process_pair(vid, kp_path, args.out_dir)
            converted.append(out_path)
            print(f"Wrote {out_path}")
        except Exception as e:
            print(f"Failed {vid}: {e}")

    print(f"Converted {len(converted)} sequences to {args.out_dir}")


if __name__ == '__main__':
    main()
