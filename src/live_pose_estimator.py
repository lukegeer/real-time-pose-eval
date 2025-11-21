from pathlib import Path
import cv2
import numpy as np
import pickle
import time
from media_pipe_pose import MediaPipePose
from util.visualize_tools import get_mp_keypoints, resize_height_and_keypoints, create_visualization
from util.pose_metrics import extract_joint_angles, get_overlay, calculate_pose_similarity, draw_limb_vectors

# Refactor it so that all the webcam body overlays
# 
# 
# 
#  keypoints, 

### CONFIGURATION ###
WEBCAM_INDEX = 1    # 1 for MacBook
CONFIDENCE_THRESHOLD = 0.5

# AIST++ reference configuration
AIST_VIDEO_PATH = "./data/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"
AIST_KEYPOINT_PATH = "./data/processed/aist_plusplus_final/keypoints2d/gBR_sBM_cAll_d04_mBR0_ch01.pkl"
AIST_START_FRAME = 0
AIST_STILL = True

# Display settings
TARGET_HEIGHT = 1080 # 720  # Resize frames to this height for consistent display


### NEW CODE: UI Helper to draw the score bar ###
def draw_score_ui(frame, score, vec_score, ang_score):
    # Color coding: Red < 50 < Yellow < 80 < Green
    if score > 80: color = (0, 255, 0)
    elif score > 50: color = (0, 255, 255)
    else: color = (0, 0, 255)

    # Draw Background Bar
    cv2.rectangle(frame, (20, 20), (320, 90), (0, 0, 0), -1)
    
    # Draw Progress Bar
    bar_width = int((score / 100) * 280)
    cv2.rectangle(frame, (30, 55), (30 + bar_width, 75), color, -1)
    cv2.rectangle(frame, (30, 55), (310, 75), (255, 255, 255), 2) # Border

    # Text stats
    cv2.putText(frame, f"MATCH: {int(score)}%", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Small debug stats below
    cv2.putText(frame, f"Angle: {int(ang_score)}% | Vector: {int(vec_score)}%", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def main():
    print("=" * 60)
    print("Real-Time Pose Tracking with Reference")
    print("=" * 60)
    
    pose_detector = MediaPipePose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=CONFIDENCE_THRESHOLD
    )

    # display settings
    show_keypoints = True
    flip_webcam = False
    aist_still_flag = True
    aist_current_frame = AIST_START_FRAME

    print("\nOpening AIST video...")
    aist_cap = cv2.VideoCapture(AIST_VIDEO_PATH)
    if not aist_cap.isOpened():
        raise ValueError(f"Could not open video: {AIST_VIDEO_PATH}")

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
            

            # read AIST frame (still or video)
            if aist_still_flag:
                # only first AIST frame if still
                if AIST_STILL:
                    aist_still_flag = False

                # seek to desired AIST frame
                aist_cap.set(cv2.CAP_PROP_POS_FRAMES, aist_current_frame)
                success, aist_frame = aist_cap.read()

                if not success:
                    raise ValueError(f"Could not read frame {aist_current_frame} from {AIST_VIDEO_PATH}")
                
                # load ground truth keypoints
                with open(AIST_KEYPOINT_PATH , 'rb') as f:
                    data = pickle.load(f)
                
                # extract AIST keypoints
                aist_gt_keypoints = data['keypoints2d'][0][aist_current_frame]
                aist_mp_keypoints = get_mp_keypoints(pose_detector, aist_frame)

                # resize AIST frame and keypoints to target height
                aist_frame, aist_mp_keypoints, aist_gt_keypoints = resize_height_and_keypoints(aist_frame, TARGET_HEIGHT, aist_mp_keypoints, aist_gt_keypoints)

                # extract AIST angles
                aist_angles = extract_joint_angles(aist_mp_keypoints, CONFIDENCE_THRESHOLD)

                aist_title = f"REFERENCE | FRAME {AIST_START_FRAME} | VIDEO ID: {Path(AIST_VIDEO_PATH).name}"
                aist_vis_frame = create_visualization(
                    aist_frame, 
                    confidence_threshold=CONFIDENCE_THRESHOLD, 
                    mp_keypoints=aist_mp_keypoints, 
                    gt_keypoints=aist_gt_keypoints, 
                    title=aist_title
                )

                # update frame counter if not still
                if not AIST_STILL:
                    aist_current_frame += 1
                    # NEED TO UPDATE LENGTH OVERFLOW TO SET A LOOP


            # read webcam
            success, webcam_frame = webcam_cap.read()
            if not success:
                print("Failed to read from webcam")
                break
            
            # flip webcam for mirror view
            if flip_webcam:
                webcam_frame = cv2.flip(webcam_frame, 1)

            # extract webcam keypoints
            webcam_mp_keypoints = get_mp_keypoints(pose_detector, webcam_frame)

            # resize webcam frame and keypoints to target height
            webcam_frame, webcam_mp_keypoints, _ = resize_height_and_keypoints(webcam_frame, TARGET_HEIGHT, webcam_mp_keypoints)

            # extract webcam angles
            webcam_angles = extract_joint_angles(webcam_mp_keypoints, CONFIDENCE_THRESHOLD)
            
            # create webcam visualization
            webcam_title = "LIVE WEBCAM"
            webcam_footer = "q: quit | k: toggle keypoints"
            if not show_keypoints:
                webcam_mp_keypoints = None
            webcam_vis_frame = create_visualization(
                webcam_frame, 
                confidence_threshold=CONFIDENCE_THRESHOLD, 
                mp_keypoints=webcam_mp_keypoints, 
                title=webcam_title,
                fps=fps,
                footer=webcam_footer
            )


            # compare Live Angles/Vectors vs Reference Angles/Vectors
            total_score, vec_score, ang_score = calculate_pose_similarity(
                aist_gt_keypoints,
                webcam_mp_keypoints,
                aist_angles,
                webcam_angles,
                conf_threshold=CONFIDENCE_THRESHOLD,
                angle_sigma=30.0,
                vector_weight=0.5,
                angle_weight=0.5
            )

            if show_keypoints:
                # 1. Draw the joint angle arcs on the live feed
                webcam_vis_frame = get_overlay(webcam_vis_frame, webcam_mp_keypoints, webcam_angles, CONFIDENCE_THRESHOLD)
                
                # 2. Optionally draw limb direction vectors (for debugging)
                # Uncomment the line below to see directional arrows
                # from util.pose_metrics import draw_limb_vectors
                # webcam_vis_frame = draw_limb_vectors(webcam_vis_frame, webcam_mp_keypoints, CONFIDENCE_THRESHOLD)
                
                # 3. Draw the score bar
                draw_score_ui(webcam_vis_frame, total_score, vec_score, ang_score)
            

            # combine frames side by side
            combined_frame = np.hstack([webcam_vis_frame, aist_vis_frame])
            
            # display
            cv2.imshow('Live vs Reference Pose Tracking', combined_frame)
            

            # handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('k'):
                show_keypoints = not show_keypoints
                print(f"Keypoints: {'ON' if show_keypoints else 'OFF'}")
    
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