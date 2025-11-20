import cv2
import numpy as np
import time
import mediapipe as mp

def dense_optical_flow_with_keypoints(method, video_path, params=[], to_gray=None, overlay=True, scale=0.25, skip_frames=0, profile=False):
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=Lite, 1=Full, 2=Heavy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return
    
    original_h, original_w = old_frame.shape[:2]
    print(f"Original frame size: {old_frame.shape[:2]}")
    
    # Resize for optical flow
    old_frame_small = cv2.resize(old_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    print(f"Resized frame size: {old_frame_small.shape[:2]} (scale={scale})")
    
    if to_gray is None:
        to_gray = (method == cv2.calcOpticalFlowFarneback)
    
    old_gray = cv2.cvtColor(old_frame_small, cv2.COLOR_BGR2GRAY) if to_gray else old_frame_small.copy()
    
    hsv = np.zeros_like(old_frame_small)
    hsv[..., 1] = 255
    flow_bgr = np.zeros_like(old_frame_small)
    
    fps_values = []
    frame_count = 0
    cv2.namedWindow("Optical Flow + Pose", cv2.WINDOW_NORMAL)

    while True:
        ret, new_frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            continue

        start = time.time()
        
        # 1. Run MediaPipe on original resolution
        results = pose.process(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
        
        # 2. Resize for optical flow
        new_frame_small = cv2.resize(new_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        new_gray = cv2.cvtColor(new_frame_small, cv2.COLOR_BGR2GRAY) if to_gray else new_frame_small.copy()

        # 3. Compute optical flow on small frames
        if callable(method):
            flow = method(old_gray, new_gray, None, *params)
        else:
            flow = method.calc(old_gray, new_gray, None)

        # 4. Visualize optical flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 90 / np.pi
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, flow_bgr)

        if overlay:
            display_frame = cv2.addWeighted(new_frame_small, 0.25, flow_bgr, 0.75, 0)
        else:
            display_frame = flow_bgr.copy()
        
        # 5. Draw keypoints - scaled coords down
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                # Convert normalized coords to pixel coords in SMALL frame
                x = int(landmark.x * original_w * scale)
                y = int(landmark.y * original_h * scale)
                
                # Draw keypoint
                if 0 <= x < display_frame.shape[1] and 0 <= y < display_frame.shape[0]:
                    cv2.circle(display_frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.circle(display_frame, (x, y), 1, (0, 0, 0), 1)
            
            # Draw skeleton connections
            mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
            )

        # FPS display
        end = time.time()
        fps_real = 1 / (end - start + 1e-6)
        fps_values.append(fps_real)
        if len(fps_values) > 30:
            fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values)
        
        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("Optical Flow + Pose", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        old_gray = new_gray

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print(f"\nAverage FPS: {avg_fps:.2f}")

def main():
    video_path = "../../data/videos/gBR_sBM_c01_d04_mBR0_ch08.mp4"

    method = cv2.calcOpticalFlowFarneback
    
    params = [
        0.5,   # pyr_scale 
        1,     # levels (single level for speed)
        8,     # winsize (small window)
        1,     # iterations (single iteration)
        5,     # poly_n 
        1.1,   # poly_sigma
        0      # flags
    ]
    
    print("Starting optical flow processing...")
    print("Press ESC to quit\n")
    

    dense_optical_flow_with_keypoints(method, video_path, params=params, overlay=True, scale=0.25, skip_frames=0, profile=True)

if __name__ == "__main__":
    main()