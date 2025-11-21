import numpy as np


class KeypointIndex:
    """AIST 17-keypoint format indices"""
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


BODY_KEYPOINTS_ONLY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] # exclude face (0-4)


### POSITION NORMALIZATION


def get_body_reference_length(keypoints, conf_threshold=0.5):
    """
    Calculate torso height as reference length for normalization.
    Uses midpoint between shoulders to midpoint between hips.
    """
    # get shoulder midpoint
    left_shoulder = keypoints[KeypointIndex.LEFT_SHOULDER]
    right_shoulder = keypoints[KeypointIndex.RIGHT_SHOULDER]
    
    if left_shoulder[2] < conf_threshold or right_shoulder[2] < conf_threshold:
        return None
    
    shoulder_mid = (left_shoulder[:2] + right_shoulder[:2]) / 2
    
    # get hip midpoint
    left_hip = keypoints[KeypointIndex.LEFT_HIP]
    right_hip = keypoints[KeypointIndex.RIGHT_HIP]
    
    if left_hip[2] < conf_threshold or right_hip[2] < conf_threshold:
        return None
        
    hip_mid = (left_hip[:2] + right_hip[:2]) / 2
    
    # calculate torso length
    torso_length = np.linalg.norm(shoulder_mid - hip_mid)
    
    # sanity check
    if torso_length < 10: # too small, probably bad detection
        return None
        
    return torso_length


def get_body_origin(keypoints, conf_threshold=0.5):
    """
    Calculate hip midpoint as the origin for normalization.
    
    Returns:
        np.array: (x, y) coordinates of hip midpoint, or None if invalid
    """
    left_hip = keypoints[KeypointIndex.LEFT_HIP]
    right_hip = keypoints[KeypointIndex.RIGHT_HIP]
    
    if left_hip[2] < conf_threshold or right_hip[2] < conf_threshold:
        return None
        
    hip_mid = (left_hip[:2] + right_hip[:2]) / 2
    return hip_mid


def normalize_keypoints(keypoints, conf_threshold=0.5):
    """
    Normalize keypoints to body-relative coordinate system.
    - Origin at hip midpoint (0, 0)
    - Scale based on torso length (torso = 1.0 unit)
    
    This makes poses comparable across different:
    - Distances from camera (scale invariant)
    - Positions in frame (translation invariant)
    
    Returns:
        np.array: Normalized keypoints (x, y, confidence) or None if failed
    """
    origin = get_body_origin(keypoints, conf_threshold)
    torso_length = get_body_reference_length(keypoints, conf_threshold)
    
    if origin is None or torso_length is None:
        return None
    
    normalized = keypoints.copy()
    
    # translate to origin and scale by torso length
    for i in range(len(normalized)):
        if normalized[i, 2] >= conf_threshold:
            # translate
            normalized[i, 0] -= origin[0]
            normalized[i, 1] -= origin[1]
            
            # scale
            normalized[i, 0] /= torso_length
            normalized[i, 1] /= torso_length
            
            # keep confidence unchanged
    
    return normalized


### SIMILARITY SCORING (position-based only)


def calculate_position_similarity(ref_keypoints, live_keypoints, conf_threshold=0.5, distance_threshold=0.2, exclude_face_from_similarity=True):
    """
    Calculate pose similarity using normalized keypoint positions.
    This metric captures position, direction, and angles all at once.
    
    Uses Gaussian-weighted scoring with asymmetric penalties for missing keypoints.
    
    Args:
        ref_keypoints: Reference pose keypoints
        live_keypoints: Live pose keypoints  
        conf_threshold: Minimum confidence for keypoint validity
        distance_threshold: Maximum normalized distance for "correct" keypoint
                           (default 0.2 = 20% of torso length)
    
    Returns:
        float: Position similarity score (0-100)
    """
    # normalize both poses
    ref_norm = normalize_keypoints(ref_keypoints, conf_threshold)
    live_norm = normalize_keypoints(live_keypoints, conf_threshold)
    
    if ref_norm is None or live_norm is None:
        return None # cannot calculate - normalization failed
    
    distance_scores = []
    
    # determine which keypoints to evaluate
    keypoint_indices = BODY_KEYPOINTS_ONLY if exclude_face_from_similarity else range(len(ref_norm))

    for i in keypoint_indices:
        ref_visible = ref_norm[i, 2] >= conf_threshold
        live_visible = live_norm[i, 2] >= conf_threshold
        
        if ref_visible and live_visible:
            dist = np.linalg.norm(ref_norm[i, :2] - live_norm[i, :2])
            score = np.exp(-(dist**2) / (2 * (distance_threshold/2)**2))
            distance_scores.append(score)
            
        elif ref_visible and not live_visible:
            distance_scores.append(0.0)
    
    if not distance_scores:
        return 0.0
    
    return np.mean(distance_scores) * 100



def calculate_per_keypoint_similarity(ref_keypoints, live_keypoints, conf_threshold=0.5, distance_threshold=0.2, exclude_face_from_similarity=True):
    """
    Calculate similarity score for each individual keypoint.
    Used for color-coding visualization.
    
    Args:
        ref_keypoints: Reference pose keypoints
        live_keypoints: Live pose keypoints
        conf_threshold: Minimum confidence threshold
        distance_threshold: Distance threshold for normalization
    
    Returns:
        dict: {keypoint_index: score (0-100)} for each valid keypoint
    """
    # normalize both poses
    ref_norm = normalize_keypoints(ref_keypoints, conf_threshold)
    live_norm = normalize_keypoints(live_keypoints, conf_threshold)
    
    if ref_norm is None or live_norm is None:
        return None # cannot calculate - normalization failed
    
    keypoint_scores = {}
    
    for i in range(len(ref_norm)):
        ref_visible = ref_norm[i, 2] >= conf_threshold
        live_visible = live_norm[i, 2] >= conf_threshold
        
        if ref_visible and live_visible:
            # both visible: calculate distance-based score
            dist = np.linalg.norm(ref_norm[i, :2] - live_norm[i, :2])
            score = np.exp(-(dist**2) / (2 * (distance_threshold/2)**2))
            keypoint_scores[i] = score * 100
            
        elif ref_visible and not live_visible:
            # reference visible but user missing: zero score
            keypoint_scores[i] = 0.0
    
    return keypoint_scores


def score_to_color(score):
    """
    Convert similarity score (0-100) to BGR color gradient.
    0-50: Red -> Yellow
    50-100: Yellow -> Green
    
    Args:
        score: Similarity score (0-100)
    
    Returns:
        tuple: BGR color (B, G, R)
    """
    score = np.clip(score, 0, 100)
    
    if score < 50:
        # Red (0,0,255) -> Yellow (0,255,255)
        # Increase green component
        ratio = score / 50.0
        b = 0
        g = int(255 * ratio)
        r = 255
    else:
        # Yellow (0,255,255) -> Green (0,255,0)
        # Decrease red component
        ratio = (score - 50) / 50.0
        b = 0
        g = 255
        r = int(255 * (1 - ratio))
    
    return (b, g, r)

