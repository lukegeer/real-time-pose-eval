import cv2
import mediapipe as mp
import numpy as np


### CONFIGURATION ###
WEBCAM_INDEX = 1  # 0 or 1 for MacBook webcam
CONFIDENCE_THRESHOLD = 0.5
SHOW_SEGMENTATION = True  # toggle body segmentation mask


# initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def main():
    pose = mp_pose.Pose(
        min_detection_confidence=CONFIDENCE_THRESHOLD,
        min_tracking_confidence=CONFIDENCE_THRESHOLD,
        enable_segmentation=SHOW_SEGMENTATION, # enable segmentation mask
        model_complexity=1 # 0=lite, 1=full, 2=heavy
    )
    
    # initialize webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    show_segmentation = SHOW_SEGMENTATION
    show_keypoints = True
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
        
        # flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # process frame
        results = pose.process(rgb_frame)
        output_frame = frame.copy()
        
        if results.pose_landmarks:
            # draw body segmentation mask
            if show_segmentation and results.segmentation_mask is not None:
                # create colored mask overlay
                segmentation_mask = results.segmentation_mask
                mask_img = np.zeros_like(frame)
                mask_img[:] = (0, 255, 0) # green color for body
                
                # apply mask with threshold
                condition = segmentation_mask > 0.5
                mask_overlay = np.where(condition[:, :, None], mask_img, 0)
                
                # blend mask with original frame
                output_frame = cv2.addWeighted(output_frame, 0.7, mask_overlay.astype(np.uint8), 0.3, 0)
            
            # draw pose landmarks and connections
            if show_keypoints:
                mp_drawing.draw_landmarks(
                    output_frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # display keypoint count
            num_visible = sum(1 for landmark in results.pose_landmarks.landmark if landmark.visibility > CONFIDENCE_THRESHOLD)
            cv2.putText(output_frame, f"Visible keypoints: {num_visible}/33", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # no pose detected
            cv2.putText(output_frame, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # display instructions
        cv2.putText(output_frame, "Press 'q' to quit | 's' for segmentation | 'k' for keypoints", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # show frame
        cv2.imshow('Pose Tracking', output_frame)
        
        # handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_segmentation = not show_segmentation
            print(f"Segmentation: {'ON' if show_segmentation else 'OFF'}")
        elif key == ord('k'):
            show_keypoints = not show_keypoints
            print(f"Keypoints: {'ON' if show_keypoints else 'OFF'}")
    
    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Pose tracking stopped")

if __name__ == "__main__":
    main()