import numpy as np
import cv2


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


# defined as (Point 1, Vertex/Center, Point 3)
ANGLE_POINTS = {
    "left_elbow":      (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),
    "right_elbow":     (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),
    "left_shoulder":   (KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
    "right_shoulder":  (KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
    "left_knee":       (KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_KNEE, KeypointIndex.LEFT_ANKLE),
    "right_knee":      (KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_KNEE, KeypointIndex.RIGHT_ANKLE),
    "left_hip":        (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_KNEE),
    "right_hip":       (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_KNEE),
}

# define limb segments for directional vectors (Start Point, End Point)
LIMB_VECTORS = {
    'torso_left':  (5, 11),  # Left Shoulder to Left Hip
    'torso_right': (6, 12),  # Right Shoulder to Right Hip
    'arm_upper_l': (5, 7),   # Left Shoulder to Elbow
    'arm_lower_l': (7, 9),   # Left Elbow to Wrist
    'arm_upper_r': (6, 8),   # Right Shoulder to Elbow
    'arm_lower_r': (8, 10),  # Right Elbow to Wrist
    'leg_upper_l': (11, 13), # Left Hip to Knee
    'leg_lower_l': (13, 15), # Left Knee to Ankle
    'leg_upper_r': (12, 14), # Right Hip to Knee
    'leg_lower_r': (14, 16), # Right Knee to Ankle
}


### ANGLE CALCULATION


def compute_angle(p1, p2, p3):
    """Computes scalar angle (magnitude) in degrees [0, 180], p2 is the vertex"""
    v1 = p1 - p2
    v2 = p3 - p2
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return np.nan
    
    # dot product formula
    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (norm1 * norm2)
    
    # numerical stability clip
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.degrees(np.arccos(cos_angle))


def extract_joint_angles(keypoints, confidence_threshold=0.5):
    """
    Extracts joint angles from keypoints based on ANGLE_POINTS dictionary.
    
    Returns:
        dict: Joint name -> angle in degrees
    """
    angles = {}
    
    for name, (idx1, idx2, idx3) in ANGLE_POINTS.items():
        # Check confidence of all three points
        if (keypoints[idx1, 2] < confidence_threshold or 
            keypoints[idx2, 2] < confidence_threshold or 
            keypoints[idx3, 2] < confidence_threshold):
            continue

        p1 = keypoints[idx1, :2]
        p2 = keypoints[idx2, :2]
        p3 = keypoints[idx3, :2]

        angle = compute_angle(p1, p2, p3)
        
        if not np.isnan(angle):
            angles[name] = angle
            
    return angles


### VECTOR CALCULATION


def get_limb_vector(keypoints, idx1, idx2, conf_threshold=0.5):
    """
    Calculates the unit vector for a limb segment.
    
    Returns:
        np.array: Unit vector from point idx1 to idx2, or None if invalid
    """
    # Check confidence
    if keypoints[idx1, 2] < conf_threshold or keypoints[idx2, 2] < conf_threshold:
        return None
        
    p1 = keypoints[idx1, :2]
    p2 = keypoints[idx2, :2]
    
    vector = p2 - p1
    norm = np.linalg.norm(vector)
    
    # Handle zero length
    if norm < 1e-6:
        return None
        
    return vector / norm



### SIMILARITY SCORING


def calculate_vector_similarity(ref_keypoints, live_keypoints, conf_threshold=0.5):
    """
    Calculate limb direction similarity using cosine similarity.
    
    Returns:
        float: Vector similarity score (0-100)
    """
    vector_scores = []
    
    for name, (start, end) in LIMB_VECTORS.items():
        v_ref = get_limb_vector(ref_keypoints, start, end, conf_threshold)
        v_live = get_limb_vector(live_keypoints, start, end, conf_threshold)
        
        if v_ref is not None and v_live is not None:
            # Dot product of unit vectors = Cosine Similarity (-1 to 1)
            cosine_sim = np.dot(v_ref, v_live)
            
            # Normalize -1..1 to 0..1 range
            # 1.0 -> 1.0 (Perfect)
            # 0.0 -> 0.5 (Perpendicular)
            # -1.0 -> 0.0 (Opposite)
            score = (cosine_sim + 1) / 2
            vector_scores.append(score)
            
    return np.mean(vector_scores) * 100 if vector_scores else 0.0


def calculate_angle_similarity(ref_angles, live_angles, sigma=30.0):
    """
    Calculate joint angle similarity using Gaussian decay.
    
    Args:
        sigma: Controls strictness (default 30 degrees)
    
    Returns:
        float: Angle similarity score (0-100)
    """
    angle_scores = []
    
    common_joints = set(ref_angles.keys()) & set(live_angles.keys())
    for name in common_joints:
        diff = abs(ref_angles[name] - live_angles[name])
        
        # Gaussian scoring: e^(-diff^2 / (2*sigma^2))
        # 0 deg diff = 1.0
        # 30 deg diff ~= 0.6
        # 60 deg diff ~= 0.13
        score = np.exp(-(diff**2) / (2 * sigma**2))
        angle_scores.append(score)
        
    return np.mean(angle_scores) * 100 if angle_scores else 0.0


def calculate_pose_similarity(ref_keypoints, live_keypoints, ref_angles, live_angles, 
                              conf_threshold=0.5, angle_sigma=30.0, 
                              vector_weight=0.5, angle_weight=0.5):
    """
    Combined pose similarity score using both limb directions and joint angles.
    
    Args:
        ref_keypoints: Reference pose keypoints
        live_keypoints: Live pose keypoints
        ref_angles: Reference joint angles dict
        live_angles: Live joint angles dict
        conf_threshold: Minimum confidence for keypoint validity
        angle_sigma: Strictness parameter for angle matching (degrees)
        vector_weight: Weight for direction component (default 0.5)
        angle_weight: Weight for angle component (default 0.5)
    
    Returns:
        tuple: (total_score, vector_score, angle_score)
    """
    vec_score = calculate_vector_similarity(ref_keypoints, live_keypoints, conf_threshold)
    ang_score = calculate_angle_similarity(ref_angles, live_angles, angle_sigma)
    
    total_score = (vector_weight * vec_score) + (angle_weight * ang_score)
    
    return total_score, vec_score, ang_score


### VISUALIZATION


def get_text_position(p2, start_angle, end_angle, distance=20):
    """
    Calculates the best position for text so it sits in the 'mouth' of the angle.
    
    Args:
        p2: The vertex point (x, y)
        start_angle: Start of the arc in degrees
        end_angle: End of the arc in degrees
        distance: How far away from the vertex to place the text
    """
    # 1. Handle the wrap-around case (e.g., arc goes from 350° to 10°)
    if end_angle < start_angle:
        end_angle += 360

    # 2. Calculate the bisector (middle angle)
    mid_angle = (start_angle + end_angle) / 2
    
    # 3. Convert back to radians for trig
    mid_rad = np.radians(mid_angle)
    
    # 4. Calculate offset
    offset_x = int(distance * np.cos(mid_rad))
    offset_y = int(distance * np.sin(mid_rad))
    
    return (int(p2[0]) + offset_x, int(p2[1]) + offset_y)


def draw_angle_arc(image, p1, p2, p3, angle_magnitude, radius=35, color=(0,255,0), thickness=2):
    """
    Draws an arc representing the interior angle and places text in the 'mouth' of the angle.
    
    Args:
        image: Canvas to draw on
        p1, p2, p3: Coordinates (p2 is the vertex)
        angle_magnitude: The scalar angle (0-180) calculated previously
    """
    # 1. Vector math to find start/end angles
    v1 = np.array(p1[:2]) - np.array(p2[:2])
    v2 = np.array(p3[:2]) - np.array(p2[:2])
    
    # Calculate absolute angles in image space (0-360 degrees)
    a1 = np.degrees(np.arctan2(v1[1], v1[0])) % 360
    a2 = np.degrees(np.arctan2(v2[1], v2[0])) % 360
    
    # 2. Logic to choose the 'Interior' Arc
    diff = (a2 - a1) % 360
    
    if abs(diff - angle_magnitude) < 5.0:
        start_angle = a1
        end_angle = a2
    else:
        start_angle = a2
        end_angle = a1

    # 3. Draw the Arc
    center = (int(p2[0]), int(p2[1]))
    cv2.ellipse(
        image, 
        center, 
        (radius, radius), 
        0, 
        start_angle, 
        end_angle, 
        color, 
        thickness, 
        cv2.LINE_AA
    )
    
    # 4. Smart Text Positioning
    if end_angle < start_angle:
        mid_angle_deg = (start_angle + (end_angle + 360)) / 2
    else:
        mid_angle_deg = (start_angle + end_angle) / 2
        
    mid_angle_rad = np.radians(mid_angle_deg)
    
    text_padding = 20 
    text_dist = radius + text_padding
    
    offset_x = int(text_dist * np.cos(mid_angle_rad))
    offset_y = int(text_dist * np.sin(mid_angle_rad))
    
    text_pos = (center[0] + offset_x, center[1] + offset_y)
    
    # 5. Draw Text (Centered)
    text_str = f"{int(angle_magnitude)}°"
    (w, h), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    final_text_pos = (text_pos[0] - w // 2, text_pos[1] + h // 2)
    
    cv2.putText(
        image,
        text_str,
        final_text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA
    )


def get_overlay(frame, keypoints, angles, confidence_threshold=0.5):
    """
    Draws angle arcs on the frame for all detected joints.
    
    Returns:
        np.array: Frame with angle overlays
    """
    overlay = frame.copy()

    for name, angle_val in angles.items():
        if name not in ANGLE_POINTS:
            continue

        i1, i2, i3 = ANGLE_POINTS[name]
        
        if (keypoints[i1, 2] < confidence_threshold or 
            keypoints[i2, 2] < confidence_threshold or 
            keypoints[i3, 2] < confidence_threshold):
            continue

        p1 = keypoints[i1]
        p2 = keypoints[i2]
        p3 = keypoints[i3]

        draw_angle_arc(
            overlay, 
            p1, p2, p3, 
            angle_magnitude=angle_val, 
            radius=30, 
            color=(0, 255, 255), 
            thickness=2
        )
    
    return overlay


def draw_limb_vectors(image, keypoints, confidence_threshold=0.5, 
                     color=(255, 100, 0), thickness=2):
    """
    Draw directional arrows for each limb segment.
    Useful for debugging direction matching.
    
    Args:
        image: Frame to draw on
        keypoints: Pose keypoints
        confidence_threshold: Minimum confidence for drawing
        color: Arrow color in BGR (default: orange)
        thickness: Arrow line thickness
    
    Returns:
        np.array: Frame with vector arrows
    """
    for name, (start, end) in LIMB_VECTORS.items():
        if (keypoints[start, 2] < confidence_threshold or 
            keypoints[end, 2] < confidence_threshold):
            continue
            
        p1 = tuple(keypoints[start, :2].astype(int))
        p2 = tuple(keypoints[end, :2].astype(int))
        
        # Draw arrow from start to end
        cv2.arrowedLine(image, p1, p2, color, thickness, tipLength=0.3)
    
    return image