import cv2
from media_pipe_pose import MediaPipePose


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


def resize_height_and_keypoints(frame, target_height, mp_keypoints=None, gt_keypoints=None):
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

    if mp_keypoints is not None:
        mp_keypoints = scale_keypoints(mp_keypoints, original_height, target_height)

    if gt_keypoints is not None:
        gt_keypoints = scale_keypoints(gt_keypoints, original_height, target_height)
    
    return frame, mp_keypoints, gt_keypoints


def draw_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, text_color=COLORS_BGR['white'], thickness=2, bg_color=None, alpha=0.6, padding=4):
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


def visualize_pose(frame, keypoints, color=COLORS_BGR['green'], confidence_threshold=0.3, draw_skeleton=True):
    """Draw pose keypoints and skeleton on frame"""
    num_visible = 0
    
    # draw keypoints
    for (x, y, conf) in keypoints:
        if conf > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
            num_visible += 1
    
    # draw skeleton
    if draw_skeleton:
        for (a, b) in LIMBS:
            if keypoints[a, 2] > confidence_threshold and keypoints[b, 2] > confidence_threshold:
                pt1 = (int(keypoints[a, 0]), int(keypoints[a, 1]))
                pt2 = (int(keypoints[b, 0]), int(keypoints[b, 1]))
                cv2.line(frame, pt1, pt2, color, 2)
    
    return num_visible


def create_visualization(frame, confidence_threshold=0.5, mp_keypoints=None, gt_keypoints=None, title=None, fps=0, footer=None):
    """Create visualization of frame with MediaPipe keypoints, and optional title, footer and ground truth keypoints"""
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

    # add title and fps if video
    if title:
        if fps > 0:
            title += f" | FPS: {fps:.1f}"
        draw_text(vis_frame, title, topleft_pos, bg_color=COLORS_BGR['black'])
        topleft_pos = move_pos(topleft_pos, diff_down)

    # draw MediaPipe predictions (green)
    if mp_keypoints is not None:
        mp_color = COLORS_BGR['green']
        num_mp = visualize_pose(vis_frame, mp_keypoints, color=mp_color, confidence_threshold=confidence_threshold)
        if num_mp > 0:
            draw_text(vis_frame, f"MP: {num_mp}/17", topleft_pos, text_color=mp_color, bg_color=COLORS_BGR['black'])
        else:
            draw_text(vis_frame, "NO POSE DETECTED", topleft_pos, text_color=mp_color, bg_color=COLORS_BGR['black'])
        topleft_pos = move_pos(topleft_pos, diff_down)

    # draw ground truth (blue)
    if gt_keypoints is not None:
        gt_color = COLORS_BGR['blue']
        num_gt = visualize_pose(vis_frame, gt_keypoints, color=gt_color, confidence_threshold=confidence_threshold)
        draw_text(vis_frame, f"GT: {num_gt}/17", topleft_pos, text_color=gt_color, bg_color=COLORS_BGR['black'])
        topleft_pos = move_pos(topleft_pos, diff_down)
    
    if mp_keypoints is None and gt_keypoints is None:
        draw_text(vis_frame, "KEYPOINTS DISABLED", topleft_pos, text_color=COLORS_BGR['red'], bg_color=COLORS_BGR['black'])
        topleft_pos = move_pos(topleft_pos, diff_down)
    
    if footer:
        draw_text(vis_frame, footer, bottomleft_pos,  bg_color=COLORS_BGR['black'])
        bottomleft_pos = move_pos(bottomleft_pos, diff_up)

    return vis_frame

