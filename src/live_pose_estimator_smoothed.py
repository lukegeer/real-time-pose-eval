from pathlib import Path
import cv2
import numpy as np
import time
import sys, os
sys.path.append(os.path.abspath(".."))
from src.model.embedding_model import PoseEmbeddingNet
from src.model.media_pipe_pose import MediaPipePose
from src.parser.live_video_parser import VideoParser
import torch
from util.visualize_tools import get_mp_keypoints, resize_height_and_keypoints, create_visualization, draw_score_ui
from util.pose_metrics import calculate_position_similarity, calculate_per_keypoint_similarity, score_to_color
from src.pose_pipeline import export_video_keypoints_to_pkl, process_frame, compute_embed, pairwise_distance_error
from src.model.kalman import create_kalman_with_acceleration


### CONFIGURATION ###
# AIST++ reference configuration
REF_VIDEO_PATH = "../data/raw/gBR/ch01/gBR_sBM_c01_d04_mBR0_ch01.mp4"
AIST_KEYPOINT_PATH = "../data/processed/aist_plusplus_final/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
AIST_START_FRAME = 0
AIST_VIDEO = False
AIST_MP_KEYPOINTS = False

# Display settings
WEBCAM_INDEX = 0    # 1 for MacBook
TARGET_HEIGHT = 1080 # Resize frames to this height for consistent display (has performance effect)
FLIP_WEBCAM = True
TARGET_FPS = 7.5

# Similarity settings
CONFIDENCE_THRESHOLD = 0.1
EXCLUDE_FACE_FROM_SIMILARITY = True

# MediaPipe pose settings
STATIC_IMAGE_MODE = False
MODEL_COMPLEXITY = 0




def main():
    print("=" * 60)
    print("Real-Time Pose Tracking with Reference")
    print("=" * 60)

    pose_detector = MediaPipePose(
        static_image_mode=STATIC_IMAGE_MODE,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=CONFIDENCE_THRESHOLD
    )
    ckpt_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    embed_path = os.path.join(ckpt_dir, "best_model.pth")
    embed_model = PoseEmbeddingNet(embed_dims=128, hidden_dims=[512, 512], dropout=0.3)
    embed_model.load_state_dict(torch.load(embed_path, map_location='cpu'))
    embed_model.eval()

    print("\nOpening AIST video...")
    ref_parser = VideoParser(REF_VIDEO_PATH)
    # Try to play reference at 60 FPS if possible
    ref_parser.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # get total AIST frame count
    total_aist_frames = int(ref_parser.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"AIST video has {total_aist_frames} frames")

    # load pickle once at startup
    print("\nLoading ground truth keypoints...")
    _, aist_data = export_video_keypoints_to_pkl(
        REF_VIDEO_PATH,
        "./data/processed/aist_plusplus_final/keypoints2d/live_ref.pkl",
        create_kalman_with_acceleration,
        target_fps=TARGET_FPS,
    )
    aist_all_keypoints = aist_data["keypoints2d"][0]
    keypoint_type = "MP" if AIST_MP_KEYPOINTS else "GT"
    print(f"Loaded {len(aist_all_keypoints)} keypoint frames")

    # set AIST starting frame
    aist_current_frame = AIST_START_FRAME

    print("\nOpening webcam...")
    webcam_parser = VideoParser(WEBCAM_INDEX, live=True)

    # get webcam dimensions
    webcam_width = int(webcam_parser.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(webcam_parser.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened: {webcam_width}x{webcam_height}")
    
    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()

    def distance_error_to_score(err, midpoint=10.0, steepness=0.3):
        """
        Map pairwise distance error (0 best, ~30 worst) to a 0-100 score via decreasing logistic.
        """
        if err is None or not np.isfinite(err):
            return 0.0
        prob = 1.0 / (1.0 + np.exp((err - midpoint) * steepness))
        return float(np.clip(prob * 100.0, 0.0, 100.0))

    def ensure_xyc(arr):
        arr = np.asarray(arr)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr

    def init_state():
        return {
            "kalman_filters": [],
            "prev_gray_small": None,
            "prev_flow": None,
            "prev_keypoints": None,
            "prev_prev_keypoints": None,
            "prev_raw": None,
            "prev_prev_raw": None,
            "prev_conf": None,
            "flow_history": None,
            "vel_state": None,
            "fallback_counts": None,
            "canonical_lengths": None,
            "bbox_state": {"ema": None, "ema_wh": None, "prev": None},
        }
    
    state = init_state()
    ref_iter = iter(ref_parser)
    web_iter = iter(webcam_parser)
    # Skip ref frames to speed playback (stride 2 -> ~2x faster)
    ref_stride = 2
    
    print("\n" + "=" * 60)
    print("Starting live tracking... Press 'q' to quit")
    print("=" * 60 + "\n")
    
    try:
        for pkt_idx in range(int(1e9)):
            try:
                ref_pkt = next(ref_iter)
                # Skip extra ref frames to speed playback
                for _ in range(ref_stride - 1):
                    next(ref_iter, None)
                web_pkt = next(web_iter)
            except StopIteration:
                break
            # calculate FPS
            ref_frame = ref_pkt["frame"]
            elapsed = time.time() - start_time
            fps = frame_count / elapsed

            # Apply same stride to reference keypoints
            key_idx = frame_count * ref_stride
            if key_idx >= len(aist_all_keypoints):
                print("Reached end of reference keypoints; stopping.")
                break

            ref_keypoints = ensure_xyc(aist_all_keypoints[key_idx])
            ref_frame, ref_keypoints = resize_height_and_keypoints(ref_frame, TARGET_HEIGHT, ref_keypoints)

            # create AIST visualization
            ref_title = f"REFERENCE | FRAME {aist_current_frame} | VIDEO ID: {Path(REF_VIDEO_PATH).name} | TYPE: {keypoint_type}"
            ref_vis_frame = create_visualization(
                ref_frame, 
                confidence_threshold=CONFIDENCE_THRESHOLD if AIST_MP_KEYPOINTS else 0.0,
                keypoints=ref_keypoints,
                title=ref_title
            )

            live_frame = web_pkt["frame"]
            if FLIP_WEBCAM:
                live_frame = cv2.flip(live_frame, 1)

            pred_raw, live_keypoints, state = process_frame(
                live_frame, state, pose_detector, create_kalman_with_acceleration, 1 / TARGET_FPS
            )

            # Debug: track finite counts through pipeline
            nan_raw = np.isnan(pred_raw).sum()
            nan_smooth = np.isnan(live_keypoints).sum()
            if nan_raw > 0 or nan_smooth > 0:
                raw_conf = pred_raw[:, 2]
                smooth_conf = live_keypoints[:, 2]
                nan_joints = np.where(
                    ~np.isfinite(live_keypoints[:, 0])
                    | ~np.isfinite(live_keypoints[:, 1])
                )[0]
                meas_sample = {int(j): pred_raw[j].tolist() for j in nan_joints[:3]} if len(nan_joints) else {}
                print(
                    f"[debug frame {frame_count}] nan_raw={nan_raw}, nan_smooth={nan_smooth}, "
                    f"raw_conf_minmax=({np.nanmin(raw_conf):.3f},{np.nanmax(raw_conf):.3f}), "
                    f"smooth_conf_minmax=({np.nanmin(smooth_conf):.3f},{np.nanmax(smooth_conf):.3f}), "
                    f"nan_joint_idxs={nan_joints.tolist() if len(nan_joints)<10 else nan_joints[:10].tolist()} "
                    f"meas_sample={meas_sample}"
                )
            
            # resize AIST frame and keypoints to target height
            live_frame, live_keypoints = resize_height_and_keypoints(live_frame, TARGET_HEIGHT, live_keypoints)

            pd_err = pairwise_distance_error(ref_keypoints, live_keypoints, conf_threshold=CONFIDENCE_THRESHOLD)
            total_score = distance_error_to_score(pd_err)

            keypoint_scores = calculate_per_keypoint_similarity(ref_keypoints, live_keypoints, conf_threshold=CONFIDENCE_THRESHOLD)

            # convert scores to colors
            if keypoint_scores is None:
                # cannot calculate - set all keypoints to white (error state)
                # use all 17 keypoint indices
                keypoint_colors = {i: (255, 255, 255) for i in range(17)}
            else:
                keypoint_colors = {idx: score_to_color(score) for idx, score in keypoint_scores.items()}
            

            # create webcam visualization
            webcam_title = "LIVE WEBCAM"
            if fps > 0:
                webcam_title += f" | FPS: {fps:.1f}"
            webcam_title += " | q: quit"
            webcam_vis_frame = create_visualization(
                live_frame, 
                confidence_threshold=CONFIDENCE_THRESHOLD, 
                keypoints=live_keypoints, 
                title=webcam_title,
                mp_keypoint_colors=keypoint_colors
            )

            draw_score_ui(webcam_vis_frame, total_score)

            # combine frames side by side
            combined_frame = np.hstack([webcam_vis_frame, ref_vis_frame])
        
            # display
            cv2.imshow('Live vs Reference Pose Tracking', combined_frame)
            frame_count += 1
            
            # handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # cleanup
        cv2.destroyAllWindows()
        print("Cleanup complete")
        print("\nFinal stats:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Average FPS: {fps:.1f}")


if __name__ == "__main__":
    main()
