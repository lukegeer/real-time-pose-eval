import cv2
import numpy as np
import time
import mediapipe as mp
from typing import Dict, List, Any
import json

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
                                       confidence_threshold=0.5, collect_stats=False,
                                       sample_grid_size=None):
    """
    Optical flow with keypoint prediction for low-confidence detections
    
    Args:
        sample_grid_size: If set, compute flow on a grid sample rather than all pixels
    
    Returns:
        stats: Dictionary with statistics if collect_stats=True, else None
    """
    # Only initialize MediaPipe if we're not just collecting stats
    if not collect_stats:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        predictor = KeypointPredictor(
            num_keypoints=33,
            confidence_threshold=confidence_threshold,
            history_size=5,
            flow_scale=scale
        )
    else:
        pose = None
        predictor = None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None

    ret, old_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame.")
        return None
    
    original_h, original_w = old_frame.shape[:2]
    if not collect_stats:
        print(f"Original frame size: ({original_h}, {original_w})")
    
    # Resize for optical flow
    old_frame_small = cv2.resize(old_frame, None, fx=scale, fy=scale, 
                                 interpolation=cv2.INTER_LINEAR)
    flow_h, flow_w = old_frame_small.shape[:2]
    if not collect_stats:
        print(f"Flow resolution: ({flow_h}, {flow_w}) (scale={scale})")
    
    if to_gray is None:
        to_gray = (method == cv2.calcOpticalFlowFarneback)
    
    old_gray = cv2.cvtColor(old_frame_small, cv2.COLOR_BGR2GRAY) if to_gray else old_frame_small.copy()
    
    hsv = np.zeros_like(old_frame_small)
    hsv[..., 1] = 255
    flow_bgr = np.zeros_like(old_frame_small)
    
    fps_values = []
    frame_count = 0
    prediction_stats = {"total": 0, "predicted": 0}
    
    # Statistics collection
    flow_magnitudes = []
    
    if not collect_stats:
        cv2.namedWindow("Optical Flow + Predicted Keypoints", cv2.WINDOW_NORMAL)

    while True:
        ret, new_frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
            continue

        start = time.time()
        
        # Resize for optical flow
        new_frame_small = cv2.resize(new_frame, None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_LINEAR)
        
        new_gray = cv2.cvtColor(new_frame_small, cv2.COLOR_BGR2GRAY) if to_gray else new_frame_small.copy()

        # Compute optical flow on small frames
        if callable(method):
            flow = method(old_gray, new_gray, None, *params)
        else:
            flow = method.calc(old_gray, new_gray, None)

        # Collect flow statistics
        if collect_stats:
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Sample grid if specified for faster processing
            if sample_grid_size is not None:
                h, w = mag.shape
                y_samples = np.linspace(0, h-1, sample_grid_size, dtype=int)
                x_samples = np.linspace(0, w-1, sample_grid_size, dtype=int)
                yy, xx = np.meshgrid(y_samples, x_samples, indexing='ij')
                mag = mag[yy, xx]
            
            flow_magnitudes.append(mag)
        else:
            # Only run MediaPipe and keypoint prediction when visualizing
            # Run MediaPipe on ORIGINAL resolution
            results = pose.process(cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB))
            
            # Update keypoints with flow prediction
            keypoints, confidences, predicted_mask = predictor.update(
                results.pose_landmarks if results else None,
                flow,
                (original_h, original_w),
                (flow_h, flow_w)
            )
            
            # Track stats
            prediction_stats["total"] += len(keypoints)
            prediction_stats["predicted"] += np.sum(predicted_mask)
        
        # Visualize only if not collecting stats
        if not collect_stats:
            if overlay:
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang * 90 / np.pi
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, flow_bgr)
                display_frame = cv2.addWeighted(new_frame_small, 0.4, flow_bgr, 0.6, 0)
            else:
                display_frame = new_frame_small.copy()
            
            # Draw keypoints with different colors for predicted vs detected
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

            cv2.imshow("Optical Flow + Predicted Keypoints", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        else:
            # Update FPS for stats
            end = time.time()
            fps_real = 1 / (end - start + 1e-6)
            fps_values.append(fps_real)

        old_gray = new_gray

    cap.release()
    if not collect_stats:
        cv2.destroyAllWindows()
        pose.close()
    
    # Calculate statistics
    pred_pct = (prediction_stats["predicted"] / max(1, prediction_stats["total"])) * 100
    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
    
    if not collect_stats:
        print(f"Keypoints predicted by flow: {pred_pct:.1f}%")
    
    if collect_stats and flow_magnitudes:
        # Compute flow statistics
        all_mags = np.concatenate([m.flatten() for m in flow_magnitudes])
        
        stats = {
            'frames_processed': len(flow_magnitudes),
            'overall_mean_magnitude': float(np.mean(all_mags)),
            'overall_median_magnitude': float(np.median(all_mags)),
            'overall_std_magnitude': float(np.std(all_mags)),
            'overall_max_magnitude': float(np.max(all_mags)),
            'percentile_95': float(np.percentile(all_mags, 95)),
            'motion_coverage_0.5': float(np.mean(all_mags > 0.5)),
            'avg_fps': float(avg_fps),
            'prediction_percentage': 0.0  # Not calculated in stats mode
        }
        return stats
    
    return None


def print_summary_report(results: Dict[int, Dict[str, Any]], failed_videos: List[int]):
    """Print comprehensive summary report."""
    
    print("\n\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    if not results:
        print("No results generated. Check file paths and data format.")
        return
    
    # Summary table
    print(f"\n{'CH':<4} {'Frames':<8} {'Mean':<8} {'Median':<8} {'Std':<8} {'Max':<8} {'P95':<8} {'Motion%':<9} {'FPS':<6}")
    print("-" * 75)
    
    for ch in sorted(results.keys()):
        m = results[ch]
        print(f"{ch:02d}   "
              f"{m['frames_processed']:<8} "
              f"{m['overall_mean_magnitude']:<8.4f} "
              f"{m['overall_median_magnitude']:<8.4f} "
              f"{m['overall_std_magnitude']:<8.4f} "
              f"{m['overall_max_magnitude']:<8.4f} "
              f"{m['percentile_95']:<8.4f} "
              f"{m['motion_coverage_0.5']:<9.2%} "
              f"{m['avg_fps']:<6.1f}")
    
    # Aggregate statistics
    if len(results) > 1:
        print("\n" + "="*80)
        print("AGGREGATE STATISTICS ACROSS ALL CHANNELS")
        print("="*80)
        
        all_means = [m['overall_mean_magnitude'] for m in results.values()]
        all_medians = [m['overall_median_magnitude'] for m in results.values()]
        all_stds = [m['overall_std_magnitude'] for m in results.values()]
        all_maxs = [m['overall_max_magnitude'] for m in results.values()]
        all_motion_coverage = [m['motion_coverage_0.5'] for m in results.values()]
        all_p95 = [m['percentile_95'] for m in results.values()]
        all_fps = [m['avg_fps'] for m in results.values()]
        
        print(f"\nMagnitude Statistics:")
        print(f"  Mean across channels: {np.mean(all_means):.4f} ± {np.std(all_means):.4f}")
        print(f"  Median across channels: {np.mean(all_medians):.4f} ± {np.std(all_medians):.4f}")
        print(f"  Std across channels: {np.mean(all_stds):.4f} ± {np.std(all_stds):.4f}")
        print(f"  Max across channels: {np.mean(all_maxs):.4f} ± {np.std(all_maxs):.4f}")
        print(f"  95th percentile: {np.mean(all_p95):.4f} ± {np.std(all_p95):.4f}")
        
        print(f"\nMotion Activity:")
        print(f"  Mean motion coverage (>0.5): {np.mean(all_motion_coverage):.2%} ± {np.std(all_motion_coverage):.2%}")
        
        print(f"\nPerformance:")
        print(f"  Average FPS: {np.mean(all_fps):.1f} ± {np.std(all_fps):.1f}")
        
        # Channel rankings
        max_motion_ch = max(results.items(), key=lambda x: x[1]['overall_mean_magnitude'])
        min_motion_ch = min(results.items(), key=lambda x: x[1]['overall_mean_magnitude'])
        
        print(f"\nChannel Rankings:")
        print(f"  Highest motion: CH {max_motion_ch[0]:02d} (mean={max_motion_ch[1]['overall_mean_magnitude']:.4f})")
        print(f"  Lowest motion: CH {min_motion_ch[0]:02d} (mean={min_motion_ch[1]['overall_mean_magnitude']:.4f})")
        
        # Success rate
        total_attempted = len(results) + len(failed_videos)
        success_rate = len(results) / total_attempted * 100 if total_attempted > 0 else 0
        print(f"\nProcessing Statistics:")
        print(f"  Successfully processed: {len(results)}/{total_attempted} ({success_rate:.1f}%)")
        if failed_videos:
            print(f"  Failed channels: {failed_videos}")


def run_evaluation_pipeline(video_pattern, channels, output_json, scale=0.25, 
                            skip_frames=0, confidence_threshold=0.5, sample_grid_size=50):
    """
    Run optical flow evaluation across multiple video channels
    
    Args:
        video_pattern: Format string with {:02d} for channel number
        channels: List of channel numbers to process
        output_json: Path to save results JSON
        scale: Optical flow resolution scale
        skip_frames: Number of frames to skip
        confidence_threshold: Threshold for keypoint prediction
        sample_grid_size: Grid size for sampling flow magnitudes (faster)
    
    Returns:
        results: Dictionary of channel -> statistics
        failed_videos: List of failed channel numbers
    """
    method = cv2.calcOpticalFlowFarneback
    
    ultra_fast_params = [
        0.5,   # pyr_scale
        2,     # levels (reduced from 3)
        15,    # winsize (reduced from 21)
        2,     # iterations (reduced from 3)
        5,     # poly_n
        1.1,   # poly_sigma
        0      # flags
    ]
    
    results = {}
    failed_videos = []
    
    print("="*80)
    print("OPTICAL FLOW EVALUATION PIPELINE")
    print("="*80)
    print(f"Processing {len(channels)} channels...")
    print(f"Scale: {scale}, Skip frames: {skip_frames}, Confidence threshold: {confidence_threshold}")
    print(f"Sample grid size: {sample_grid_size}x{sample_grid_size}")
    print()
    
    for i, ch in enumerate(channels, 1):
        video_path = video_pattern.format(ch)
        print(f"[{i}/{len(channels)}] Channel {ch:02d}...", end=" ", flush=True)
        
        try:
            stats = dense_optical_flow_with_prediction(
                method, 
                video_path, 
                params=ultra_fast_params, 
                overlay=False,
                scale=scale, 
                skip_frames=skip_frames,
                confidence_threshold=confidence_threshold,
                collect_stats=True,
                sample_grid_size=sample_grid_size
            )
            
            if stats is not None:
                results[ch] = stats
                print(f"✓ ({stats['frames_processed']} frames, {stats['avg_fps']:.1f} fps)")
            else:
                failed_videos.append(ch)
                print("✗ Failed")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            failed_videos.append(ch)
    
    # Save results to JSON
    if results:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_json}")
    
    return results, failed_videos


def main():
    # Single video mode
    single_video = False
    
    if single_video:
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
            overlay=False,
            scale=0.25, 
            skip_frames=0,
            confidence_threshold=0.5
        )
    else:
        # Batch evaluation mode
        video_pattern = "../../data/videos/gBR_sBM_c01_d04_mBR0_ch{:02d}.mp4"
        channels = list(range(1, 11))  # Process channels 01-10
        output_json = "optical_flow_results.json"
        
        # Run pipeline
        results, failed_videos = run_evaluation_pipeline(
            video_pattern=video_pattern,
            channels=channels,
            output_json=output_json,
            scale=0.25,
            skip_frames=2,  # Skip 2 frames = process every 3rd frame for speed
            confidence_threshold=0.5,
            sample_grid_size=50  # Sample 50x50 grid instead of full resolution
        )
        
        # Print summary
        print_summary_report(results, failed_videos)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)


if __name__ == "__main__":
    main()