import cv2
import numpy as np
import time
import mediapipe as mp

class KeypointPredictor:
    """Predicts occluded keypoints using optical flow"""
    
    def __init__(self, num_keypoints=33, confidence_threshold=0.5, history_size=5, flow_scale=1.0):
        """
        Args:
            num_keypoints: Number of pose keypoints (33 for MediaPipe)
            confidence_threshold: Below this, use optical flow prediction
            history_size: Number of frames to keep for smoothing
            flow_scale: Ratio of optical flow resolution to original resolution
        """
        self.num_keypoints = num_keypoints
        self.confidence_threshold = confidence_threshold
        self.history_size = history_size
        self.flow_scale = flow_scale
        
        # Store previous keypoint positions and confidences
        self.prev_keypoints = None  # Shape: (num_keypoints, 2) in original resolution
        self.prev_confidences = None
        self.keypoint_history = []  # For smoothing predictions
        
    def update(self, landmarks, flow, original_shape, flow_shape):
        """
        Update keypoints using MediaPipe results and optical flow
        
        Args:
            landmarks: MediaPipe pose_landmarks (or None)
            flow: Optical flow field (H_flow, W_flow, 2)
            original_shape: (H, W) of original video frame
            flow_shape: (H_flow, W_flow) of optical flow computation
            
        Returns:
            keypoints: (num_keypoints, 2) array of (x, y) in original resolution
            confidences: (num_keypoints,) array of confidence scores
            predicted_mask: (num_keypoints,) boolean array, True where predicted by flow
        """
        H_orig, W_orig = original_shape
        H_flow, W_flow = flow_shape
        
        # Initialize arrays
        current_keypoints = np.zeros((self.num_keypoints, 2))
        current_confidences = np.zeros(self.num_keypoints)
        predicted_mask = np.zeros(self.num_keypoints, dtype=bool)
        
        # Extract MediaPipe keypoints if available
        if landmarks is not None:
            for i, landmark in enumerate(landmarks.landmark):
                current_keypoints[i] = [landmark.x * W_orig, landmark.y * H_orig]
                current_confidences[i] = landmark.visibility  # or use landmark.presence
        
        # If we have previous keypoints, predict using optical flow
        if self.prev_keypoints is not None:
            for i in range(self.num_keypoints):
                # Use optical flow if confidence is low or keypoint is missing
                if current_confidences[i] < self.confidence_threshold:
                    predicted_mask[i] = True
                    
                    # Get previous position
                    prev_x, prev_y = self.prev_keypoints[i]
                    
                    # Skip if previous position was invalid
                    if prev_x <= 0 or prev_y <= 0:
                        continue
                    
                    # Map to flow resolution
                    flow_x = int(prev_x * H_flow / H_orig)
                    flow_y = int(prev_y * W_flow / W_orig)
                    
                    # Check bounds
                    if 0 <= flow_y < H_flow and 0 <= flow_x < W_flow:
                        # Get flow vector at previous keypoint location
                        # Note: flow is (H, W, 2) where flow[:,:,0] is dx, flow[:,:,1] is dy
                        dx = flow[flow_y, flow_x, 0]
                        dy = flow[flow_y, flow_x, 1]
                        
                        # Scale flow back to original resolution
                        dx_orig = dx * W_orig / W_flow
                        dy_orig = dy * H_orig / H_flow
                        
                        # Predict new position
                        pred_x = prev_x + dx_orig
                        pred_y = prev_y + dy_orig
                        
                        # Clamp to image bounds
                        pred_x = np.clip(pred_x, 0, W_orig - 1)
                        pred_y = np.clip(pred_y, 0, H_orig - 1)
                        
                        current_keypoints[i] = [pred_x, pred_y]
                        # Assign reduced confidence to flow predictions
                        current_confidences[i] = self.prev_confidences[i] * 0.8
        
        # Store for next iteration
        self.prev_keypoints = current_keypoints.copy()
        self.prev_confidences = current_confidences.copy()
        
        # Smooth predictions using history
        self.keypoint_history.append(current_keypoints.copy())
        if len(self.keypoint_history) > self.history_size:
            self.keypoint_history.pop(0)
        
        # Apply temporal smoothing for predicted keypoints
        if len(self.keypoint_history) > 1:
            for i in range(self.num_keypoints):
                if predicted_mask[i]:
                    # Average over history
                    history_array = np.array([h[i] for h in self.keypoint_history])
                    current_keypoints[i] = np.mean(history_array, axis=0)
        
        return current_keypoints, current_confidences, predicted_mask


def dense_optical_flow_with_prediction(method, video_path, params=[], to_gray=None, 
                                       overlay=True, scale=0.25, skip_frames=0,
                                       confidence_threshold=0.5):
    """
    Optical flow with keypoint prediction for low-confidence detections
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
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
    print(f"Original frame size: ({original_h}, {original_w})")
    
    # Resize for optical flow
    old_frame_small = cv2.resize(old_frame, None, fx=scale, fy=scale, 
                                 interpolation=cv2.INTER_LINEAR)
    flow_h, flow_w = old_frame_small.shape[:2]
    print(f"Flow resolution: ({flow_h}, {flow_w}) (scale={scale})")
    
    if to_gray is None:
        to_gray = (method == cv2.calcOpticalFlowFarneback)
    
    old_gray = cv2.cvtColor(old_frame_small, cv2.COLOR_BGR2GRAY) if to_gray else old_frame_small.copy()
    
    hsv = np.zeros_like(old_frame_small)
    hsv[..., 1] = 255
    flow_bgr = np.zeros_like(old_frame_small)
    
    # Initialize predictor
    predictor = KeypointPredictor(
        num_keypoints=33,
        confidence_threshold=confidence_threshold,
        history_size=5,
        flow_scale=scale
    )
    
    fps_values = []
    frame_count = 0
    prediction_stats = {"total": 0, "predicted": 0}
    
    cv2.namedWindow("Optical Flow + Predicted Keypoints", cv2.WINDOW_NORMAL)

    while True:
        ret, new_frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            continue

        start = time.time()
        
        # 1. Run MediaPipe on ORIGINAL resolution
        results = pose.process(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
        
        # 2. Resize for optical flow
        new_frame_small = cv2.resize(new_frame, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_LINEAR)
        
        new_gray = cv2.cvtColor(new_frame_small, cv2.COLOR_BGR2GRAY) if to_gray else new_frame_small.copy()

        # 3. Compute optical flow on small frames
        if callable(method):
            flow = method(old_gray, new_gray, None, *params)
        else:
            flow = method.calc(old_gray, new_gray, None)

        # 4. Update keypoints with flow prediction
        keypoints, confidences, predicted_mask = predictor.update(
            results.pose_landmarks if results else None,
            flow,
            (original_h, original_w),
            (flow_h, flow_w)
        )
        
        # Track stats
        prediction_stats["total"] += len(keypoints)
        prediction_stats["predicted"] += np.sum(predicted_mask)
        
        # 5. Visualize optical flow (only if overlay is enabled)
        if overlay:
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 90 / np.pi
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, flow_bgr)
            display_frame = cv2.addWeighted(new_frame_small, 0.4, flow_bgr, 0.6, 0)
        else:
            display_frame = new_frame_small.copy()
        
        # 6. Draw keypoints with different colors for predicted vs detected
        for i, (kp, conf, is_predicted) in enumerate(zip(keypoints, confidences, predicted_mask)):
            if conf > 0.1:  # Only draw if somewhat confident
                # Scale down to display resolution
                x = int(kp[0] * scale)
                y = int(kp[1] * scale)
                
                if 0 <= x < display_frame.shape[1] and 0 <= y < display_frame.shape[0]:
                    if is_predicted:
                        # PREDICTED by optical flow - RED
                        display_frame[y, x] = [0, 0, 255]
                    else:
                        # DETECTED by MediaPipe - GREEN
                        display_frame[y, x] = [0, 255, 0]

        # FPS and stats display
        end = time.time()
        fps_real = 1 / (end - start + 1e-6)
        fps_values.append(fps_real)
        if len(fps_values) > 30:
            fps_values.pop(0)
        avg_fps = sum(fps_values) / len(fps_values)
        
        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Show prediction percentage
        pred_pct = (prediction_stats["predicted"] / max(1, prediction_stats["total"])) * 100
        cv2.putText(display_frame, f"Flow Pred: {pred_pct:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Legend
        cv2.putText(display_frame, "Green=Detected Red=Predicted", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Optical Flow + Predicted Keypoints", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        old_gray = new_gray

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    
    print(f"\nAverage FPS: {avg_fps:.2f}")
    print(f"Keypoints predicted by flow: {pred_pct:.1f}%")


def main():
    video_path = "../../data/videos/gBR_sBM_c01_d04_mBR0_ch08.mp4"

    method = cv2.calcOpticalFlowFarneback
    
    ultra_fast_params = [
        0.5,   # pyr_scale
        1,     # levels
        8,     # winsize
        1,     # iterations
        5,     # poly_n
        1.1,   # poly_sigma
        0      # flags
    ]
    
    print("Starting optical flow with keypoint prediction...")
    print("Green dots = MediaPipe detected")
    print("Red dots = Optical flow predicted")
    print("Press ESC to quit\n")
    
    dense_optical_flow_with_prediction(
        method, video_path, 
        params=ultra_fast_params, 
        overlay=False,  # Set to True to see optical flow
        scale=0.25, 
        skip_frames=0,
        confidence_threshold=0.5  # Predict keypoints below 50% confidence
    )


if __name__ == "__main__":
    main()