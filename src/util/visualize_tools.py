import cv2


# skeleton connections for 17-keypoint AIST format
LIMBS = [
    (5,7),(7,9),(6,8),(8,10),
    (11,13),(13,15),(12,14),(14,16),
    (5,6),(11,12),(5,11),(6,12)
]


# color definitions (BGR format)
COLORS_BGR = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'magenta': (255, 0, 255),
    'cyan': (255, 255, 0),
    'gray': (128, 128, 128),
}


def get_mp_keypoints(pose_detector, frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_keypoints_33 = pose_detector.detect_landmarks(rgb_frame)
    mp_keypoints = pose_detector.convert_to_aist17(mp_keypoints_33)
    return mp_keypoints


def resize_height_and_keypoints(frame, target_height, keypoints=None):
    def resize_to_height(frame, target_height):
        """Resize frame to target height while maintaining aspect ratio."""
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(frame, (target_width, target_height))

    def scale_keypoints(keypoints, original_height, target_height):
        """Scale keypoint coordinates proportionally when frame is resized."""
        scale_factor = target_height / original_height
        scaled_keypoints = keypoints.copy()
        scaled_keypoints[:, 0] *= scale_factor  # scale x coordinates
        scaled_keypoints[:, 1] *= scale_factor  # scale y coordinates
        return scaled_keypoints

    # resize reference to target height
    original_height = frame.shape[0]
    frame = resize_to_height(frame, target_height)

    if keypoints is not None:
        keypoints = scale_keypoints(keypoints, original_height, target_height)
    
    return frame, keypoints


def draw_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, text_color=COLORS_BGR['white'], thickness=2, bg_color=None, alpha=0.6, padding=4):
    """Draw text with optional semi-transparent background"""
    # add optional background
    if bg_color is not None:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = position
        
        rect_x1 = max(0, x - padding)
        rect_y1 = max(0, y - text_height - padding)
        rect_x2 = min(frame.shape[1], x + text_width + padding)
        rect_y2 = min(frame.shape[0], y + baseline + padding)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # add text
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)


def visualize_pose(frame, keypoints, color=COLORS_BGR['green'], confidence_threshold=0.3, draw_skeleton=True, keypoint_colors=None):
    """Draw pose keypoints and skeleton on frame"""
    num_visible = 0
    
    # draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            # Use per-keypoint color if provided, otherwise use default
            kp_color = keypoint_colors.get(i, color) if keypoint_colors else color
            cv2.circle(frame, (int(x), int(y)), 5, kp_color, -1)
            num_visible += 1
    
    # draw skeleton
    if draw_skeleton:
        for (a, b) in LIMBS:
            if keypoints[a, 2] > confidence_threshold and keypoints[b, 2] > confidence_threshold:
                pt1 = (int(keypoints[a, 0]), int(keypoints[a, 1]))
                pt2 = (int(keypoints[b, 0]), int(keypoints[b, 1]))
                
                # Use average color of the two endpoint keypoints for the limb
                if keypoint_colors:
                    color_a = keypoint_colors.get(a, color)
                    color_b = keypoint_colors.get(b, color)
                    # Average the two colors
                    limb_color = tuple(int((color_a[i] + color_b[i]) / 2) for i in range(3))
                else:
                    limb_color = color
                
                cv2.line(frame, pt1, pt2, limb_color, 2)
    
    return num_visible


def create_visualization(frame, confidence_threshold=0.5, keypoints=None, title=None, mp_keypoint_colors=None):
    """Create visualization of frame, with optional keypoints and title"""
    vis_frame = frame.copy()
    h = vis_frame.shape[0]
    
    # label positions
    def move_pos(current_pos, diff_pos):
        new_pos = tuple(a + b for a, b in zip(current_pos, diff_pos))
        return new_pos
    topleft_pos = (10, 20)
    bottomleft_pos = (10, h-20)
    diff_down = (0, 30)
    diff_up = (0, -30)

    # add title
    if title:
        draw_text(vis_frame, title, topleft_pos, bg_color=COLORS_BGR['black'])
        topleft_pos = move_pos(topleft_pos, diff_down)

    # draw keypoints (with dynamic coloring if provided)
    if keypoints is not None:
        color = COLORS_BGR['green']
        num_mp = visualize_pose(vis_frame, keypoints, color=color, 
                               confidence_threshold=confidence_threshold,
                               keypoint_colors=mp_keypoint_colors)
        if num_mp > 0:
            draw_text(vis_frame, f"Keypoints: {num_mp}/17", topleft_pos, text_color=color, bg_color=COLORS_BGR['black'])
        else:
            draw_text(vis_frame, "NO POSE DETECTED", topleft_pos, text_color=color, bg_color=COLORS_BGR['black'])
        topleft_pos = move_pos(topleft_pos, diff_down)
    else:
        draw_text(vis_frame, "KEYPOINTS DISABLED", topleft_pos, text_color=COLORS_BGR['red'], bg_color=COLORS_BGR['black'])
        topleft_pos = move_pos(topleft_pos, diff_down)

    return vis_frame


def draw_score_ui(frame, score):
    # box dimensions
    box_width = 300
    box_height = 90
    
    # reference point (top-right corner of frame)
    x_ref = frame.shape[1] - box_width - 4
    y_ref = 4
    
    # draw background box
    cv2.rectangle(frame, (x_ref, y_ref), (x_ref + box_width, y_ref + box_height), (0, 0, 0), -1)
    
    if score is None:
        # cannot calculate similarity - show warning
        cv2.putText(frame, "CANNOT CALCULATE", 
                   (x_ref + 10, y_ref + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2) # orange
        cv2.putText(frame, "Hips not detected", 
                   (x_ref + 10, y_ref + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, "Step back from camera", 
                   (x_ref + 10, y_ref + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    else:
        # color coding: red < 50 < yellow < 80 < green
        if score > 80: color = (0, 255, 0)
        elif score > 50: color = (0, 255, 255)
        else: color = (0, 0, 255)
        
        # progress bar dimensions
        bar_x = x_ref + 10
        bar_y = y_ref + 35
        bar_max_width = 280
        bar_height = 20
        
        # draw progress bar
        bar_width = int((score / 100) * bar_max_width)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_width, bar_y + bar_height), (255, 255, 255), 2)  # Border

        # text stats
        cv2.putText(frame, f"MATCH: {int(score)}%", 
                   (bar_x, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # small descriptor below
        cv2.putText(frame, "Position-based similarity", 
                   (bar_x, bar_y + bar_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
