from pathlib import Path
import cv2
import numpy as np
import pickle
import time
from media_pipe_pose import MediaPipePose
from util.visualize_tools import get_mp_keypoints, resize_height_and_keypoints, create_visualization, draw_score_ui
from util.pose_metrics import calculate_position_similarity, calculate_per_keypoint_similarity, score_to_color


### CONFIGURATION ###
# AIST++ reference configuration
AIST_VIDEO_PATH = "./data/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"
AIST_KEYPOINT_PATH = "./data/processed/aist_plusplus_final/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
AIST_START_FRAME = 0
AIST_VIDEO = False
AIST_MP_KEYPOINTS = False

# Display settings
WEBCAM_INDEX = 1    # 1 for MacBook
TARGET_HEIGHT = 1080 # Resize frames to this height for consistent display (has performance effect)
FLIP_WEBCAM = True

# Similarity settings
CONFIDENCE_THRESHOLD = 0.5
EXCLUDE_FACE_FROM_SIMILARITY = False

# MediaPipe pose settings
STATIC_IMAGE_MODE = False
MODEL_COMPLEXITY = 1




def main():
    print("=" * 60)
    print("Real-Time Pose Tracking with Reference")
    print("=" * 60)

    pose_detector = MediaPipePose(
        static_image_mode=STATIC_IMAGE_MODE,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=CONFIDENCE_THRESHOLD
    )

    print("\nOpening AIST video...")
    aist_cap = cv2.VideoCapture(AIST_VIDEO_PATH)
    if not aist_cap.isOpened():
        raise ValueError(f"Could not open video: {AIST_VIDEO_PATH}")

    # get total AIST frame count
    total_aist_frames = int(aist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"AIST video has {total_aist_frames} frames")

    # load pickle once at startup
    print("\nLoading ground truth keypoints...")
    with open(AIST_KEYPOINT_PATH, 'rb') as f:
        aist_data = pickle.load(f)
    aist_all_keypoints = aist_data['keypoints2d'][0]
    print(f"Loaded {len(aist_all_keypoints)} keypoint frames")

    # set AIST starting frame
    aist_current_frame = AIST_START_FRAME

    print("\nOpening webcam...")
    webcam_cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not webcam_cap.isOpened():
        print("Error: Could not open webcam")
        return

    # get webcam dimensions
    webcam_width = int(webcam_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(webcam_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam opened: {webcam_width}x{webcam_height}")
    
    # FPS calculation
    fps = 0
    frame_count = 0
    fps_update_interval = 10 # update FPS every N frames
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print("Starting live tracking... Press 'q' to quit")
    print("=" * 60 + "\n")
    
    try:
        while webcam_cap.isOpened():
            # calculate FPS periodically
            frame_count += 1
            if frame_count % fps_update_interval == 0:
                elapsed = time.time() - start_time
                fps = fps_update_interval / elapsed
                start_time = time.time()
            
            # only read/process if in video mode OR if this is the first frame in still mode
            if AIST_VIDEO or frame_count == 1:
                # read AIST frame
                aist_cap.set(cv2.CAP_PROP_POS_FRAMES, aist_current_frame)
                success, aist_frame = aist_cap.read()

                if not success:
                    # loop back to start if we've reached the end
                    aist_current_frame = 0
                    aist_cap.set(cv2.CAP_PROP_POS_FRAMES, aist_current_frame)
                    success, aist_frame = aist_cap.read()
                    if not success:
                        raise ValueError(f"Could not read frame {aist_current_frame} from {AIST_VIDEO_PATH}")

                # extract AIST keypoints
                if AIST_MP_KEYPOINTS: # MediaPipe keypoints
                    keypoint_type = "MP"
                    aist_keypoints = get_mp_keypoints(pose_detector, aist_frame)
                else: # ground truth keypoints
                    keypoint_type = "GT"
                    aist_keypoints = aist_all_keypoints[aist_current_frame]

                # resize AIST frame and keypoints to target height
                aist_frame, aist_keypoints = resize_height_and_keypoints(aist_frame, TARGET_HEIGHT, aist_keypoints)

                # create AIST visualization
                aist_title = f"REFERENCE | FRAME {aist_current_frame} | VIDEO ID: {Path(AIST_VIDEO_PATH).name} | TYPE: {keypoint_type}"
                aist_vis_frame = create_visualization(
                    aist_frame, 
                    confidence_threshold=CONFIDENCE_THRESHOLD if AIST_MP_KEYPOINTS else 0.0,
                    keypoints=aist_keypoints,
                    title=aist_title
                )

                # advance to next frame (only if in video mode)
                if AIST_VIDEO:
                    aist_current_frame = (aist_current_frame + 1) % total_aist_frames


            # read webcam
            success, webcam_frame = webcam_cap.read()
            if not success:
                print("Failed to read from webcam")
                break
            
            # flip webcam for mirror view
            if FLIP_WEBCAM:
                webcam_frame = cv2.flip(webcam_frame, 1)

            # extract webcam keypoints
            webcam_keypoints = get_mp_keypoints(pose_detector, webcam_frame)

            # resize webcam frame and keypoints to target height
            webcam_frame, webcam_keypoints = resize_height_and_keypoints(webcam_frame, TARGET_HEIGHT, webcam_keypoints)

            # compare pose positions
            total_score = calculate_position_similarity(
                aist_keypoints,
                webcam_keypoints,
                conf_threshold=CONFIDENCE_THRESHOLD,
                exclude_face_from_similarity=EXCLUDE_FACE_FROM_SIMILARITY
            )

            # calculate per-keypoint scores for visualization coloring
            keypoint_scores = calculate_per_keypoint_similarity(
                aist_keypoints,
                webcam_keypoints,
                conf_threshold=CONFIDENCE_THRESHOLD,
                distance_threshold=0.2,
                exclude_face_from_similarity=EXCLUDE_FACE_FROM_SIMILARITY
            )

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
                webcam_frame, 
                confidence_threshold=CONFIDENCE_THRESHOLD, 
                keypoints=webcam_keypoints, 
                title=webcam_title,
                mp_keypoint_colors=keypoint_colors
            )

            draw_score_ui(webcam_vis_frame, total_score)

            # combine frames side by side
            combined_frame = np.hstack([webcam_vis_frame, aist_vis_frame])
        
            # display
            cv2.imshow('Live vs Reference Pose Tracking', combined_frame)
            
            # handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # cleanup
        aist_cap.release()
        webcam_cap.release()
        cv2.destroyAllWindows()
        pose_detector.pose.close()
        print("Cleanup complete")
        print("\nFinal stats:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Average FPS: {fps:.1f}")


if __name__ == "__main__":
    main()