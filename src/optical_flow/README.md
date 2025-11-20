# Optical Flow Keypoint Prediction

A real-time system that integrates dense optical flow with MediaPipe pose estimation to predict occluded or low-confidence keypoints using motion information.

## Overview

This system addresses a common challenge in pose estimation: when body keypoints become occluded or are detected with low confidence, traditional pose estimators struggle to track them. By leveraging dense optical flow, we can predict the position of missing keypoints based on motion patterns from previous frames.

## Methodology

Code: [Optical Flow Keypoint Prediction](optical_flow_keypoint_prediction.py)

Dense optical flow (specifically using the Farneback method) computes motion vectors for every pixel between consecutive frames. Since dense optical flow is computationally intensive, we implemented optimizations including resolution reduction and frame skipping to maintain real-time performance. This represents a typical performance-robustness tradeoff common in real-time computer vision applications.

The system processes video at two different resolutions to balance accuracy and speed:
- **Original Resolution (1920×1080)**: MediaPipe pose estimation runs on full-resolution frames to maximize keypoint detection accuracy
- **Downscaled Resolution (480×270)**: Optical flow computation runs on reduced resolution (downscaled 4x) to enable low-latency real-time performance

MediaPipe detects 33 body keypoints with associated confidence scores. When keypoints are visible and unoccluded, MediaPipe provides accurate localization. However, when body parts are:
- Occluded by other body parts or objects
- Out of frame boundaries
- In poor lighting conditions
- Moving at high velocities

... confidence scores drop below reliable thresholds, and detections may become inaccurate or missing entirely. The primary challenges observed in the AIST dataset are occlusion and rapid motion. When a keypoint's confidence falls below the threshold (default: 0.5), the system predicts its new position using optical flow.

### Prediction Procedure

1. Retrieve the keypoint's previous position in original resolution
2. Map the keypoint location to corresponding coordinates in the flow field
3. Extract the flow vector at that location
4. Scale the flow vector back to original resolution
5. Predict the new keypoint position by applying the scaled flow vector

Note: For keypoints occluded over prolonged periods, prediction confidence decays based on spatial and temporal linearity assumptions.

### Confidence Model
```
if mediapipe_confidence >= threshold:
    use MediaPipe detection
else:
    use optical flow prediction
    if optical_flow_confidence < minimum_threshold (0.1):
        stop tracking this keypoint
```

## Temporal Smoothing

We implemented temporal smoothing to reduce jitter in keypoint predictions. The system maintains information from the last N frames (default: 5) and applies temporal averaging. This smoothing is exclusively applied to keypoints predicted by optical flow, not to MediaPipe's direct detections.

## Limitations

The optical flow-based prediction approach has several limitations:

- **Error Accumulation**: Predictions may drift over time without ground truth corrections
- **Resolution Trade-off**: Flow accuracy is compromised by resolution downscaling
- **Strong Assumptions**: Predictions rely on spatial and temporal locality assumptions
- **No Semantic Understanding**: Optical flow lacks semantic comprehension of human anatomy
- **Confidence Decay**: The system eventually loses track of long-term occluded keypoints

## Future Improvements

- **Kalman Filter**: Implement better motion prediction with acceleration modeling
- **Pose Priors**: Incorporate anatomical constraints to ensure physically plausible predictions
- **Adaptive Thresholds**: Develop per-keypoint confidence thresholds through learning approaches