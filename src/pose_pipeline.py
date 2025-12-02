import sys, os
sys.path.append(os.path.abspath(".."))
import os, json, pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
from collections import deque
from src.model.media_pipe_pose import MediaPipePose
from src.model.embedding_model import PoseEmbeddingNet
from src.model.learned_kalman import LearnedKalmanDynamics
from src.util.visualizer import visualize_multiple, visualize_pose
from src.parser.live_video_parser import VideoParser
from src.util.evaluation import compute_joint_angles
from IPython.display import display, clear_output

# Displacement threshold for flagging jumpy measurements in smoothing
MAX_DISP_PX = 80.0
# Max flow-based nudge (pixels) applied during occlusion/jumps
MAX_NUDGE_PX = 5.0

def make_update_input(xy, flow):
    x_flow = int(round(xy[0]))
    y_flow = int(round(xy[1]))
    x_flow = np.clip(x_flow, 0, flow.shape[1] - 1)
    y_flow = np.clip(y_flow, 0, flow.shape[0] - 1)
    vx, vy = flow[y_flow, x_flow]
    update_input = np.array([xy[0], xy[1], vx, vy])
    return update_input

def smooth_flow(
    kalman_filters,
    num_joints,
    pred_keypoints,
    flow,
    prev_keypoints=None,
    prev_prev_keypoints=None,
    prev_raw=None,
    prev_prev_raw=None,
    vel_state=None,
    fallback_counts=None,
    dt=1/30.0,
    conf_threshold=0.5,
):
    """
    Minimal smoothing: trust raw when confidence is good; otherwise hold last smoothed position.
    No flow-based extrapolation.
    """
    if fallback_counts is None:
        fallback_counts = [0 for _ in range(num_joints)]

    smoothed_keypoints = np.zeros_like(pred_keypoints)
    for joint_idx in range(num_joints):
        kalman = kalman_filters[joint_idx]
        xy = pred_keypoints[joint_idx][:2]
        conf = pred_keypoints[joint_idx][2]
        prev_xy = prev_keypoints[joint_idx][:2] if prev_keypoints is not None else None

        kalman.predict(np.zeros(2))

        use_meas = np.all(np.isfinite(xy)) and np.isfinite(conf) and conf >= conf_threshold

        if use_meas:
            meas = np.array([xy[0], xy[1], 0.0, 0.0])
            state_upd = kalman.update(meas)
            smoothed_keypoints[joint_idx, :2] = state_upd[:2]
            smoothed_keypoints[joint_idx, 2] = conf
            fallback_counts[joint_idx] = 0
        elif prev_xy is not None and np.all(np.isfinite(prev_xy)):
            # Hold last smoothed position
            smoothed_keypoints[joint_idx, :2] = prev_xy
            smoothed_keypoints[joint_idx, 2] = 0.0
            fallback_counts[joint_idx] += 1
        else:
            smoothed_keypoints[joint_idx, :] = np.array([np.nan, np.nan, 0.0])
            fallback_counts[joint_idx] += 1

    return smoothed_keypoints, None, fallback_counts

def smooth_flow_control_only(kalman_filters, num_joints, pred_keypoints, flow):
    smoothed_keypoints = np.zeros_like(pred_keypoints)
    for joint_idx in range(num_joints):
        kalman = kalman_filters[joint_idx]
        xy = pred_keypoints[joint_idx][:2]
        if np.any(np.isnan(xy)):
            continue
        update_input = make_update_input(xy, flow)  # [x, y, vx, vy]
        vx, vy = update_input[2], update_input[3]
        kalman.predict([vx, vy])           # control = flow velocity
        xy_smoothed = kalman.update(xy)    # measurement = position only
        smoothed_keypoints[joint_idx, :2] = xy_smoothed[:2]
        smoothed_keypoints[joint_idx, 2] = pred_keypoints[joint_idx][2]
    return smoothed_keypoints

def smooth_classic(kalman_filters, num_joints, pred_keypoints):
    smoothed_keypoints = np.zeros_like(pred_keypoints)
    for joint_idx in range(num_joints):
        kalman = kalman_filters[joint_idx]
        xy = pred_keypoints[joint_idx][:2]
        if np.any(np.isnan(xy)):
            continue
        kalman.predict(np.zeros(2))          # no control input
        xy_smoothed = kalman.update(xy)      # measurement is just [x, y]
        smoothed_keypoints[joint_idx, :2] = xy_smoothed[:2]
        smoothed_keypoints[joint_idx, 2] = pred_keypoints[joint_idx][2]

    return smoothed_keypoints

def compute_accel(flow_t, flow_t_minus_1, y, x, dt):
    h, w = flow_t.shape[:2]
    # ensure indices are valid ints
    x = int(np.clip(round(x), 0, w - 1))
    y = int(np.clip(round(y), 0, h - 1))
    vx_t_minus_1, vy_t_minus_1 = flow_t_minus_1[y, x]
    vx_t, vy_t = flow_t[y, x]

    ax = (vx_t - vx_t_minus_1) / max(dt, 1e-6)
    ay = (vy_t - vy_t_minus_1) / max(dt, 1e-6)

    accel = np.array([ax, ay])

    return accel


def make_update_input_plus_accel(xy, prev_xy, flow, prev_flow, dt):
    update_input = make_update_input(xy, flow)
    if prev_flow is None or prev_xy is None:
        accel = np.zeros(2, dtype=np.float32)
    else:
        accel = compute_accel(flow, prev_flow, prev_xy[1], prev_xy[0], dt)
    return np.concatenate([update_input, accel])

def smooth_flow_with_accel(
    kalman_filters,
    prev_keypoints,
    num_joints,
    pred_keypoints,
    flow,
    prev_flow,
    dt,
    conf_threshold=0.6,
    vel_state=None,
    alpha=0.7,
    max_disp_px=MAX_DISP_PX,
    prev_prev_keypoints=None,
    prev_raw=None,
    prev_prev_raw=None,
    fallback_counts=None,
    max_vel_px=30.0,
):
    """
    Minimal variant: use raw when confident; otherwise hold last smoothed with a tiny velocity update.
    Flow is ignored for position, only used for velocity measurement when available.
    """
    if vel_state is None:
        vel_state = [np.zeros(2, dtype=np.float32) for _ in range(num_joints)]
    if fallback_counts is None:
        fallback_counts = [0 for _ in range(num_joints)]

    smoothed_keypoints = np.zeros_like(pred_keypoints)
    for joint_idx in range(num_joints):
        kalman = kalman_filters[joint_idx]
        xy = pred_keypoints[joint_idx][:2]
        prev_xy = prev_keypoints[joint_idx][:2] if prev_keypoints is not None else None
        conf = pred_keypoints[joint_idx][2]

        # Predict
        state_pred = kalman.predict(np.zeros(4))

        use_meas = np.all(np.isfinite(xy)) and np.isfinite(conf) and conf >= conf_threshold

        if use_meas:
            # Ignore flow for position; set velocities to zero
            meas = np.array([xy[0], xy[1], 0.0, 0.0, 0.0, 0.0])
            state_upd = kalman.update(meas)
            smoothed_keypoints[joint_idx, :2] = state_upd[:2]
            smoothed_keypoints[joint_idx, 2] = conf
            fallback_counts[joint_idx] = 0
        elif prev_xy is not None and np.all(np.isfinite(prev_xy)):
            # Hold last smoothed position; decay velocity
            fallback_counts[joint_idx] += 1
            v = state_pred[2:4] * (0.9 ** fallback_counts[joint_idx])
            est_xy = prev_xy + v
            meas = np.array([est_xy[0], est_xy[1], 0.0, 0.0, 0.0, 0.0])
            state_upd = kalman.update(meas)
            smoothed_keypoints[joint_idx, :2] = state_upd[:2]
            smoothed_keypoints[joint_idx, 2] = 0.0
        else:
            smoothed_keypoints[joint_idx, :] = np.array([np.nan, np.nan, 0.0])
            fallback_counts[joint_idx] += 1

    return smoothed_keypoints, vel_state, fallback_counts


def apply_learned_dynamics(learned_model, history_deque, raw_keypoints, width, height, conf_threshold=0.6):
    """
    Apply learned dynamics to predict next keypoints. Normalizes to [0,1], builds 6D state
    (x, y, vx, vy, ax, ay), runs the model, and blends prediction with raw based on confidence.
    """
    if learned_model is None or history_deque is None:
        return raw_keypoints

    xy = raw_keypoints[:, :2]
    conf = raw_keypoints[:, 2]
    xy_norm = np.stack([xy[:, 0] / width, xy[:, 1] / height], axis=-1)

    if history_deque:
        prev = history_deque[-1]
        vx = xy_norm - prev[:, :2]
        if len(history_deque) > 1:
            prev_prev = history_deque[-2]
            ax = vx - (prev[:, :2] - prev_prev[:, :2])
        else:
            ax = np.zeros_like(xy_norm)
    else:
        vx = np.zeros_like(xy_norm)
        ax = np.zeros_like(xy_norm)

    state = np.concatenate([xy_norm, vx, ax], axis=-1)
    history_deque.append(state)
    if len(history_deque) < learned_model.history:
        return raw_keypoints

    seq = torch.tensor(np.stack(list(history_deque)), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred, _ = learned_model(seq)
    pred_np = pred[0].numpy()  # [J, 6]
    xy_pred = pred_np[:, :2] * np.array([width, height])
    blended_xy = np.where(conf[:, None] > conf_threshold, xy, xy_pred)

    out = raw_keypoints.copy()
    out[:, :2] = blended_xy
    return out


LIMBS = [
    # Arms
    (5, 7),   # left_shoulder -> left_elbow
    (7, 9),   # left_elbow -> left_wrist
    (6, 8),   # right_shoulder -> right_elbow
    (8, 10),  # right_elbow -> right_wrist

    # Legs
    (11, 13), # left_hip -> left_knee
    (13, 15), # left_knee -> left_ankle
    (12, 14), # right_hip -> right_knee
    (14, 16), # right_knee -> right_ankle

    # Torso
    (5, 6),   # left_shoulder -> right_shoulder
    (11, 12), # left_hip -> right_hip
    (5, 11),  # left_shoulder -> left_hip
    (6, 12),  # right_shoulder -> right_hip
]

def compute_canonical_lengths(keypoints3d):
    canonical_lengths = np.zeros((len(LIMBS)))
    for i, (a, b) in enumerate(LIMBS):
        joint_a = keypoints3d[a, :3]
        joint_b = keypoints3d[b, :3]
        canonical_lengths[i] = np.linalg.norm(joint_b - joint_a)
    return canonical_lengths


def _valid_lengths(lengths):
    return lengths is not None and np.all(np.isfinite(lengths)) and np.nanmean(lengths) > 1e-6


def _enforce_if_valid(kp_world, canonical_lengths, min_limb=1e-6, tolerance=1e-3):
    """
    Enforce constraints if lengths are valid and world coords are finite; skip only the limbs that are degenerate.
    """
    if not _valid_lengths(canonical_lengths):
        return kp_world
    if not np.all(np.isfinite(kp_world[:, :3])):
        return kp_world
    # Skip limbs whose current length is near zero
    kp_copy = kp_world.copy()
    for i, (a, b) in enumerate(LIMBS):
        joint_a = kp_world[a, :3]
        joint_b = kp_world[b, :3]
        current_len = np.linalg.norm(joint_b - joint_a)
        if current_len < min_limb:
            continue
    return enforce_physical_constraints(kp_copy, canonical_lengths, tolerance=tolerance)

def enforce_physical_constraints(keypoints3d, canonical_lengths, tolerance=1e-3):
    keypoints3d = keypoints3d.copy()
    for i, (a, b) in enumerate(LIMBS):
        joint_a = keypoints3d[a, :3]
        joint_b = keypoints3d[b, :3]
        conf_a = keypoints3d[a, 3]
        conf_b = keypoints3d[b, 3]
        vec = joint_b - joint_a
        current_length = np.linalg.norm(vec)
        if current_length < 1e-6:
            continue  # avoid division by zero
        desired_length = canonical_lengths[i]
        if abs(current_length - desired_length) > tolerance:
            midpoint = (joint_a + joint_b) / 2
            direction = vec / current_length
            offset = direction * (desired_length / 2)
            total_conf = conf_a + conf_b + 1e-6  # avoid division by zero
            wa = 1 - (conf_a / total_conf)
            wb = 1 - (conf_b / total_conf)
            keypoints3d[a, :3] = joint_a + wa * (midpoint - offset - joint_a)
            keypoints3d[b, :3] = joint_b + wb * (midpoint + offset - joint_b)
    return keypoints3d

def weak_perspective_project(world_xy, img_xy, conf, min_joints=4, vis_threshold=0.3):
    """
    Fit isotropic scale + translation from world XY to image-normalized xy.
    Falls back to img_xy when not enough confident points are available.
    """
    mask = conf > vis_threshold
    if np.sum(mask) < min_joints:
        return img_xy

    wx, wy = world_xy[mask].T
    ux, uy = img_xy[mask].T

    wx_mean, wy_mean = wx.mean(), wy.mean()
    ux_mean, uy_mean = ux.mean(), uy.mean()

    wxc, wyc = wx - wx_mean, wy - wy_mean
    uxc, uyc = ux - ux_mean, uy - uy_mean

    denom = (wxc**2 + wyc**2).sum()
    if denom < 1e-8:
        return img_xy

    s = (wxc * uxc + wyc * uyc).sum() / denom
    tx = ux_mean - s * wx_mean
    ty = uy_mean - s * wy_mean

    proj = np.stack([s * world_xy[:, 0] + tx,
                     s * world_xy[:, 1] + ty], axis=1)
    return proj

def project_with_fallback(world_xy, img_xy, conf, min_joints=4, vis_threshold=0.3, min_spread=1e-3):
    proj = weak_perspective_project(world_xy, img_xy, conf, min_joints=min_joints, vis_threshold=vis_threshold)
    if not np.all(np.isfinite(proj)):
        return img_xy
    spread = np.linalg.norm(np.ptp(proj, axis=0))
    if spread < min_spread:
        return img_xy
    return proj

def mediapipe_3d_to_2d(keypoints3d, width, height):
    x = keypoints3d[:, 0] * width
    y = keypoints3d[:, 1] * height
    conf = keypoints3d[:, 3]
    return np.stack([x, y, conf], axis=1)


def compute_frame_metrics(pred_keypoints2d, smoothed_keypoints, gt_keypoints, embed_model):
    def safe_cosine(a, b):
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return None
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return None
        return float(np.dot(a, b) / (na * nb))

    try:
        # Joint angles
        angles_pred = compute_joint_angles(pred_keypoints2d)
        angles_gt = compute_joint_angles(gt_keypoints)
        angle_raw = safe_cosine(angles_pred, angles_gt)

        angle_smooth = None
        if np.all(np.isfinite(smoothed_keypoints)):
            angles_smooth = compute_joint_angles(smoothed_keypoints)
            angle_smooth = safe_cosine(angles_smooth, angles_gt)

        # Embeddings
        with torch.no_grad():
            emb_pred = embed_model(torch.tensor(pred_keypoints2d, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            emb_gt = embed_model(torch.tensor(gt_keypoints, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            emb_smooth = None
            if np.all(np.isfinite(smoothed_keypoints)):
                emb_smooth = embed_model(torch.tensor(smoothed_keypoints, dtype=torch.float32).unsqueeze(0)).numpy()[0]

        embed_raw = safe_cosine(emb_pred, emb_gt)
        embed_smooth = safe_cosine(emb_smooth, emb_gt) if emb_smooth is not None else None

        return angle_raw, embed_raw, angle_smooth, embed_smooth
    except Exception:
        return None, None, None, None

# Simple match thresholds; adjust as needed for your data.
ANGLE_MATCH_THR = 0.9
EMBED_MATCH_THR = 0.8
JUMP_STD_FACTOR = 3.0
JUMP_MIN_PX = 10.0
# If a measurement jumps more than this many pixels from the previous smoothed point,
# treat it as unreliable and fall back to flow-based extrapolation.
MAX_DISP_PX = 80.0


def compute_pair_metrics(pred_a, smooth_a, pred_b, smooth_b, embed_model):
    """
    Compare two predicted poses (raw and smoothed) via joint-angle and embedding cosine similarity.
    Returns (angle_raw, embed_raw, angle_smooth, embed_smooth).
    """

    def safe_cosine(a, b):
        if a is None or b is None:
            return None
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return None
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return None
        return float(np.dot(a, b) / (na * nb))

    try:
        angles_a = compute_joint_angles(pred_a)
        angles_b = compute_joint_angles(pred_b)
        angle_raw = safe_cosine(angles_a, angles_b)

        angle_smooth = None
        if np.all(np.isfinite(smooth_a)) and np.all(np.isfinite(smooth_b)):
            angles_sa = compute_joint_angles(smooth_a)
            angles_sb = compute_joint_angles(smooth_b)
            angle_smooth = safe_cosine(angles_sa, angles_sb)

        with torch.no_grad():
            emb_a = embed_model(torch.tensor(pred_a, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            emb_b = embed_model(torch.tensor(pred_b, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            emb_sa = emb_sb = None
            if np.all(np.isfinite(smooth_a)) and np.all(np.isfinite(smooth_b)):
                emb_sa = embed_model(torch.tensor(smooth_a, dtype=torch.float32).unsqueeze(0)).numpy()[0]
                emb_sb = embed_model(torch.tensor(smooth_b, dtype=torch.float32).unsqueeze(0)).numpy()[0]

        embed_raw = safe_cosine(emb_a, emb_b)
        embed_smooth = safe_cosine(emb_sa, emb_sb) if emb_sa is not None and emb_sb is not None else None

        return angle_raw, embed_raw, angle_smooth, embed_smooth
    except Exception:
        return None, None, None, None


def update_jump_stats(state, smoothed, frame_idx, stream_label):
    """
    Track per-joint displacement jumps between frames. Maintains running mean/variance per joint
    and returns a list of events where displacement exceeds mean + JUMP_STD_FACTOR * std and JUMP_MIN_PX.
    """
    events = []
    if smoothed is None or smoothed.size == 0:
        return events

    if state.get("prev_keypoints") is None:
        state["prev_keypoints"] = smoothed
        return events

    if state.get("disp_mean") is None:
        num_joints = smoothed.shape[0]
        state["disp_mean"] = np.zeros(num_joints, dtype=float)
        state["disp_m2"] = np.zeros(num_joints, dtype=float)
        state["disp_count"] = np.zeros(num_joints, dtype=int)

    prev = state["prev_keypoints"]
    disp = np.linalg.norm(smoothed[:, :2] - prev[:, :2], axis=1)

    for j, d in enumerate(disp):
        if not np.isfinite(d):
            continue
        c = state["disp_count"][j]
        mean = state["disp_mean"][j]
        m2 = state["disp_m2"][j]

        c_new = c + 1
        delta = d - mean
        mean_new = mean + delta / c_new
        m2_new = m2 + delta * (d - mean_new)

        state["disp_count"][j] = c_new
        state["disp_mean"][j] = mean_new
        state["disp_m2"][j] = m2_new

        if c > 1:
            var = m2 / max(c - 1, 1)
            std = np.sqrt(var)
            if d > mean + JUMP_STD_FACTOR * std and d > JUMP_MIN_PX:
                events.append((d, frame_idx, stream_label, j, mean, std))

    state["prev_keypoints"] = smoothed
    return events


def run_pose_pipeline(video_path, keypoint_path, out_path, create_kalman, canonical_lengths, live: bool):
    
    with open(keypoint_path, 'rb') as f:
        data = pickle.load(f)

    frame_keypoints = data['keypoints2d'][0]

    print(type(data))

    print(list(data.keys()))

    parser = VideoParser(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = 60
    dt = 1 / fps
    w = int(parser.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(parser.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"height: {h}")
    print(f"width: {w}")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    model = MediaPipePose(static_image_mode=False, model_complexity=0)
    # Pose embedding model for runtime evaluation
    embed_model = PoseEmbeddingNet(embed_dims=128, hidden_dims=[512, 512], dropout=0.3)
    embed_model.load_state_dict(torch.load('../checkpoints/best_model.pth', map_location='cpu'))
    embed_model.eval()

    # initialize kalman filter

    num_joints = frame_keypoints.shape[1]

    kalman_filters = []
    vel_state = None
    fallback_counts = None
    learned_dyn = None
    learned_history = None

    if canonical_lengths is None or len(canonical_lengths) != len(LIMBS):
        canonical_lengths = None

    for joint_idx in range(num_joints):
        x_init = frame_keypoints[0][joint_idx][0]
        y_init = frame_keypoints[0][joint_idx][1]

        try:
            kalman = create_kalman(x_init, y_init, dt=dt)
        except TypeError:
            kalman = create_kalman(x_init, y_init)

        kalman_filters.append(kalman)

    if os.path.exists('checkpoints/learned_kalman.pth'):
        try:
            learned_dyn = LearnedKalmanDynamics(num_joints=num_joints, hidden_size=16, history=3, dt=dt)
            learned_dyn.load_state_dict(torch.load('checkpoints/learned_kalman.pth', map_location='cpu'))
            learned_dyn.eval()
            learned_history = deque(maxlen=learned_dyn.history)
        except Exception:
            learned_dyn = None

    frame_idx = 0
    start_time = time.time()
    # timing accumulators
    time_detect = time_flow = time_post = time_viz = 0.0
    # evaluation accumulators
    angle_scores = []
    embed_scores = []
    angle_scores_smooth = []
    embed_scores_smooth = []
    embed_raw_match_count = 0
    embed_smooth_match_count = 0
    angle_raw_match_count = 0
    angle_smooth_match_count = 0
    match_total = 0
    mse_raw_list = []
    mse_smooth_list = []

    smoothed_keypoints = np.zeros_like(frame_keypoints[0])
    prev_gray_small = None
    prev_flow = None
    prev_xy = None
    prev_keypoints = None
    learned_history = []
    prev_prev_keypoints = None
    prev_raw = None
    prev_prev_raw = None

    for keypoints, pkt in zip(frame_keypoints, parser):
        frame = pkt['frame']
        gray_full = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        t_loop_start = time.time()

        t0 = time.time()
        # Compute optical flow on downsampled grayscale for speed
        scale = 0.5
        gray_small = cv2.resize(gray_full, (0, 0), fx=scale, fy=scale)

        if prev_gray_small is not None and not frame_idx % 2:
            flow_small = cv2.calcOpticalFlowFarneback(prev_gray_small, gray_small, None,
                                                      pyr_scale=0.5, levels=2, winsize=9,
                                                      iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
            flow = cv2.resize(flow_small, (gray_full.shape[1], gray_full.shape[0]))
            flow *= (1.0 / scale)
        else:
            flow = np.zeros((gray_full.shape[0], gray_full.shape[1], 2), dtype=np.float32)
        t_flow_end = time.time()

        t_detect_start = time.time()
        kp_img33, kp_world33 = model.detect_landmarks(frame)
        kp_img = model.convert_to_aist17(kp_img33)
        kp_world = model.convert_to_aist17_world(kp_world33)
        t_detect_end = time.time()

        if canonical_lengths is None:
            if np.all(np.isfinite(kp_world[:, :3])) and np.any(kp_world[:, :3]):
                candidate_lengths = compute_canonical_lengths(kp_world)
                if _valid_lengths(candidate_lengths):
                    canonical_lengths = candidate_lengths

        kp_world = _enforce_if_valid(kp_world, canonical_lengths)

        pred_keypoints2d = np.stack([kp_img[:, 0] * frame.shape[1],
                                     kp_img[:, 1] * frame.shape[0],
                                     kp_img[:, 3]], axis=1)

        # Apply learned dynamics to raw preds (optional)
        if learned_dyn is not None:
            pred_keypoints2d = apply_learned_dynamics(
                learned_dyn, learned_history, pred_keypoints2d, frame.shape[1], frame.shape[0]
            )
        
        # Choose smoother based on Kalman state dimension
        if kalman_filters and kalman_filters[0].x.shape[0] == 6:
            smoothed_keypoints, vel_state, fallback_counts = smooth_flow_with_accel(
                kalman_filters, prev_keypoints, num_joints, pred_keypoints2d, flow, prev_flow, dt, vel_state=vel_state, prev_prev_keypoints=prev_prev_keypoints, prev_raw=prev_raw, prev_prev_raw=prev_prev_raw, fallback_counts=fallback_counts
            )
        elif kalman_filters[0].R.shape[0] == 4:
            smoothed_keypoints, vel_state, fallback_counts = smooth_flow(
                kalman_filters, num_joints, pred_keypoints2d, flow, prev_keypoints=prev_keypoints, prev_prev_keypoints=prev_prev_keypoints, prev_raw=prev_raw, prev_prev_raw=prev_prev_raw, vel_state=vel_state, fallback_counts=fallback_counts, dt=dt
            )
        else: 
            smoothed_keypoints = smooth_classic(kalman_filters, num_joints, pred_keypoints2d)

        t_post = time.time()

        angle_raw, embed_raw, angle_smooth, embed_smooth = compute_frame_metrics(
            pred_keypoints2d, smoothed_keypoints, keypoints, embed_model
        )
        if angle_raw is not None:
            angle_scores.append(angle_raw)
        if embed_raw is not None:
            embed_scores.append(embed_raw)
        if angle_smooth is not None:
            angle_scores_smooth.append(angle_smooth)
        if embed_smooth is not None:
            embed_scores_smooth.append(embed_smooth)

        if angle_raw is not None and embed_raw is not None:
            match_total += 1
            if embed_raw >= EMBED_MATCH_THR:
                embed_raw_match_count += 1
            if angle_raw >= ANGLE_MATCH_THR:
                angle_raw_match_count += 1
        if angle_smooth is not None and embed_smooth is not None:
            if embed_smooth >= EMBED_MATCH_THR:
                embed_smooth_match_count += 1
            if angle_smooth >= ANGLE_MATCH_THR:
                angle_smooth_match_count += 1

        # Per-frame MSE (xy only) against ground truth
        if np.all(np.isfinite(pred_keypoints2d)) and np.all(np.isfinite(keypoints)):
            mse_raw = np.mean((pred_keypoints2d[:, :2] - keypoints[:, :2]) ** 2)
            mse_raw_list.append(mse_raw)
        if np.all(np.isfinite(smoothed_keypoints)) and np.all(np.isfinite(keypoints)):
            mse_smooth = np.mean((smoothed_keypoints[:, :2] - keypoints[:, :2]) ** 2)
            mse_smooth_list.append(mse_smooth)

        prev_flow = flow
        prev_gray_small = gray_small
        prev_prev_keypoints = prev_keypoints
        prev_keypoints = smoothed_keypoints
        prev_prev_raw = prev_raw
        prev_raw = pred_keypoints2d
        elapsed = time.time() - start_time
        current_fps = frame_idx // elapsed if elapsed > 0 else 0
        
        frame = visualize_multiple(frame, smoothed_keypoints, keypoints, fps=current_fps)
        # frame = visualize_pose(frame, pred_keypoints2d, fps=current_fps, color=(0,0,255))
        t_viz = time.time()

        time_flow += (t_flow_end - t0)
        time_detect += (t_detect_end - t_detect_start)
        time_post += (t_post - t_detect_end)
        time_viz += (t_viz - t_post)
        if frame_idx and frame_idx % 60 == 0:
            frames = frame_idx or 1
            print(f"[timing] avg detect: {time_detect/frames:.4f}s, flow: {time_flow/frames:.4f}s, post: {time_post/frames:.4f}s, viz: {time_viz/frames:.4f}s")

        if live:
            plt.axis('off')
            plt.imshow(frame)
            display(plt.gcf())
            clear_output(wait=True)

        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    writer.release()

    total_time = time.time() - start_time
    final_fps = frame_idx / total_time

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Total frames: {frame_idx}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average processing FPS: {final_fps:.1f}")
    print(f"Real-time factor: {final_fps/fps:.2f}x")
    print(f"Target FPS: {fps}")
    if angle_scores:
        print(f"Avg joint-angle similarity (raw):     {np.nanmean(angle_scores):.3f}")
    else:
        print("Avg joint-angle similarity (raw):     n/a (no valid frames)")
    if angle_scores_smooth:
        print(f"Avg joint-angle similarity (smoothed): {np.nanmean(angle_scores_smooth):.3f}")
    else:
        print("Avg joint-angle similarity (smoothed): n/a (no valid frames)")
    if embed_scores:
        print(f"Avg embedding similarity (raw):       {np.nanmean(embed_scores):.3f}")
    else:
        print("Avg embedding similarity (raw):       n/a (no valid frames)")
    if embed_scores_smooth:
        print(f"Avg embedding similarity (smoothed):   {np.nanmean(embed_scores_smooth):.3f}")
    else:
        print("Avg embedding similarity (smoothed):   n/a (no valid frames)")
    if match_total:
        print(f"Angle raw match frames:     {angle_raw_match_count}/{match_total} ({angle_raw_match_count/max(match_total,1):.2f})")
    else:
        print("Angle raw match frames:     n/a")
    if match_total:
        print(f"Angle smoothed match frames:{angle_smooth_match_count}/{match_total} ({angle_smooth_match_count/max(match_total,1):.2f})")
    else:
        print("Angle smoothed match frames:n/a")
    if match_total:
        print(f"Embed raw match frames:     {embed_raw_match_count}/{match_total} ({embed_raw_match_count/max(match_total,1):.2f})")
    else:
        print("Embed raw match frames:     n/a")
    if match_total:
        print(f"Embed smoothed match frames:{embed_smooth_match_count}/{match_total} ({embed_smooth_match_count/max(match_total,1):.2f})")
    else:
        print("Embed smoothed match frames:n/a")
    if mse_raw_list:
        print(f"Avg MSE (raw xy):           {np.mean(mse_raw_list):.2f}")
    else:
        print("Avg MSE (raw xy):           n/a")
    if mse_smooth_list:
        print(f"Avg MSE (smoothed xy):      {np.mean(mse_smooth_list):.2f}")
    else:
        print("Avg MSE (smoothed xy):      n/a")


def run_dual_pose_pipeline(video_path_a, video_path_b, out_path_a, out_path_b, create_kalman, canonical_lengths=None, live=False, target_fps=30):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = target_fps
    dt = 1 / fps

    parser_a = VideoParser(video_path_a)
    parser_b = VideoParser(video_path_b)

    w_a = int(parser_a.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_a = int(parser_a.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_b = int(parser_b.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_b = int(parser_b.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer_a = cv2.VideoWriter(out_path_a, fourcc, fps, (w_a, h_a))
    writer_b = cv2.VideoWriter(out_path_b, fourcc, fps, (w_b, h_b))

    model = MediaPipePose(static_image_mode=False, model_complexity=0)
    embed_model = PoseEmbeddingNet(embed_dims=128, hidden_dims=[512, 512], dropout=0.3)
    embed_model.load_state_dict(torch.load('../checkpoints/best_model.pth', map_location='cpu'))
    embed_model.eval()
    learned_dyn = None
    if os.path.exists('../checkpoints/learned_kalman.pth'):
        try:
            learned_dyn = LearnedKalmanDynamics(num_joints=17, hidden_size=16, history=3, dt=dt)
            learned_dyn.load_state_dict(torch.load('../checkpoints/learned_kalman.pth', map_location='cpu'))
            learned_dyn.eval()
        except Exception:
            learned_dyn = None

    def init_state():
        return {
            "kalman_filters": [],
            "prev_gray_small": None,
            "prev_flow": None,
            "prev_keypoints": None,
            "prev_prev_keypoints": None,
            "prev_raw": None,
            "prev_prev_raw": None,
            "vel_state": None,
            "fallback_counts": None,
            "canonical_lengths": canonical_lengths if canonical_lengths is not None and len(canonical_lengths) == len(LIMBS) else None,
            "disp_mean": None,
            "disp_m2": None,
            "disp_count": None,
            "learned_history": deque(maxlen=learned_dyn.history) if learned_dyn else None,
        }

    state_a = init_state()
    state_b = init_state()

    angle_raw_list = []
    angle_smooth_list = []
    embed_raw_list = []
    embed_smooth_list = []
    worst_angle_raw = (np.inf, None)
    worst_angle_smooth = (np.inf, None)
    worst_raw_frames = None  # (frame_idx, score, frame_a_vis, frame_b_vis)
    worst_smooth_frames = None
    worst_jumps = []  # list of (disp, frame_idx, stream_label, joint_idx, mean, std, frame_vis)

    start_time = time.time()
    frame_idx = 0

    def process_frame(frame, state):
        gray_full = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        scale = 0.5
        gray_small = cv2.resize(gray_full, (0, 0), fx=scale, fy=scale)

        if state["prev_gray_small"] is not None and not frame_idx % 2:
            flow_small = cv2.calcOpticalFlowFarneback(state["prev_gray_small"], gray_small, None,
                                                      pyr_scale=0.5, levels=2, winsize=9,
                                                      iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
            flow = cv2.resize(flow_small, (gray_full.shape[1], gray_full.shape[0]))
            flow *= (1.0 / scale)
        else:
            flow = np.zeros((gray_full.shape[0], gray_full.shape[1], 2), dtype=np.float32)

        kp_img33, kp_world33 = model.detect_landmarks(frame)
        kp_img = model.convert_to_aist17(kp_img33)
        kp_world = model.convert_to_aist17_world(kp_world33)

        
        if state["canonical_lengths"] is None:
            if np.all(np.isfinite(kp_world[:, :3])) and np.any(kp_world[:, :3]):
                candidate_lengths = compute_canonical_lengths(kp_world)
                if _valid_lengths(candidate_lengths):
                    state["canonical_lengths"] = candidate_lengths

        kp_world = _enforce_if_valid(kp_world, state["canonical_lengths"])
        

        pred_keypoints2d = np.stack([kp_img[:, 0] * frame.shape[1],
                                     kp_img[:, 1] * frame.shape[0],
                                     kp_img[:, 3]], axis=1)

        if learned_dyn is not None:
            pred_keypoints2d = apply_learned_dynamics(
                learned_dyn, state["learned_history"], pred_keypoints2d, frame.shape[1], frame.shape[0]
            )

        if not state["kalman_filters"]:
            for joint_idx in range(pred_keypoints2d.shape[0]):
                x_init, y_init = pred_keypoints2d[joint_idx][:2]
                try:
                    kf = create_kalman(x_init, y_init, dt=dt)
                except TypeError:
                    kf = create_kalman(x_init, y_init)
                state["kalman_filters"].append(kf)

        if state["kalman_filters"] and state["kalman_filters"][0].x.shape[0] == 6:
            smoothed, state["vel_state"], state["fallback_counts"] = smooth_flow_with_accel(
                state["kalman_filters"], state["prev_keypoints"], pred_keypoints2d.shape[0], pred_keypoints2d, flow, state["prev_flow"], dt,
                vel_state=state["vel_state"], prev_prev_keypoints=state["prev_prev_keypoints"], prev_raw=state["prev_raw"], prev_prev_raw=state["prev_prev_raw"],
                fallback_counts=state["fallback_counts"]
            )
        else:
            smoothed, state["vel_state"], state["fallback_counts"] = smooth_flow(
                state["kalman_filters"], pred_keypoints2d.shape[0], pred_keypoints2d, flow,
                prev_keypoints=state["prev_keypoints"], prev_prev_keypoints=state["prev_prev_keypoints"], prev_raw=state["prev_raw"], prev_prev_raw=state["prev_prev_raw"],
                vel_state=state["vel_state"], fallback_counts=state["fallback_counts"], dt=dt
            )

        # Jump detection on smoothed keypoints
        jumps_local = update_jump_stats(state, smoothed, frame_idx, stream_label="A" if state is state_a else "B")

        state["prev_flow"] = flow
        state["prev_gray_small"] = gray_small
        state["prev_prev_keypoints"] = state["prev_keypoints"]
        state["prev_keypoints"] = smoothed
        state["prev_prev_raw"] = state["prev_raw"]
        state["prev_raw"] = pred_keypoints2d

        return pred_keypoints2d, smoothed, jumps_local

    for pkt_a, pkt_b in zip(parser_a, parser_b):
        frame_a = pkt_a['frame']
        frame_b = pkt_b['frame']

        pred_a, smooth_a, jumps_a = process_frame(frame_a, state_a)
        pred_b, smooth_b, jumps_b = process_frame(frame_b, state_b)

        angle_raw, embed_raw, angle_smooth, embed_smooth = compute_pair_metrics(pred_a, smooth_a, pred_b, smooth_b, embed_model)
        raw_is_worst = smooth_is_worst = False
        if angle_raw is not None:
            angle_raw_list.append(angle_raw)
            if angle_raw < worst_angle_raw[0]:
                worst_angle_raw = (angle_raw, frame_idx)
                raw_is_worst = True
        if angle_smooth is not None:
            angle_smooth_list.append(angle_smooth)
            if angle_smooth < worst_angle_smooth[0]:
                worst_angle_smooth = (angle_smooth, frame_idx)
                smooth_is_worst = True
        if embed_raw is not None:
            embed_raw_list.append(embed_raw)
        if embed_smooth is not None:
            embed_smooth_list.append(embed_smooth)

        elapsed = time.time() - start_time
        current_fps = frame_idx / elapsed if elapsed > 0 else 0

        frame_a_vis = visualize_multiple(frame_a.copy(), pred_a, smooth_a, fps=current_fps)
        frame_b_vis = visualize_multiple(frame_b.copy(), pred_b, smooth_b, fps=current_fps)

        # Accumulate jump events with frame snapshots
        if jumps_a:
            for ev in jumps_a:
                d, fidx, stream, joint, mean, std = ev
                worst_jumps.append((d, fidx, stream, joint, mean, std, frame_a_vis.copy()))
        if jumps_b:
            for ev in jumps_b:
                d, fidx, stream, joint, mean, std = ev
                worst_jumps.append((d, fidx, stream, joint, mean, std, frame_b_vis.copy()))
        worst_jumps = sorted(worst_jumps, key=lambda x: x[0], reverse=True)[:5]

        if raw_is_worst:
            worst_raw_frames = (frame_idx, angle_raw, frame_a_vis.copy(), frame_b_vis.copy())
        if smooth_is_worst:
            worst_smooth_frames = (frame_idx, angle_smooth, frame_a_vis.copy(), frame_b_vis.copy())

        if live:
            plt.axis('off')
            plt.imshow(np.hstack([frame_a_vis, frame_b_vis]))
            display(plt.gcf())
            clear_output(wait=True)

        writer_a.write(cv2.cvtColor(frame_a_vis, cv2.COLOR_RGB2BGR))
        writer_b.write(cv2.cvtColor(frame_b_vis, cv2.COLOR_RGB2BGR))

        frame_idx += 1

    writer_a.release()
    writer_b.release()

    total_time = time.time() - start_time
    final_fps = frame_idx / total_time if total_time > 0 else 0

    print(f"\n{'='*50}")
    print("DUAL VIDEO RESULTS")
    print(f"{'='*50}")
    print(f"Total frames processed: {frame_idx}")
    print(f"Target FPS: {fps}")
    print(f"Effective processing FPS: {final_fps:.1f}")
    if angle_raw_list:
        print(f"Avg joint-angle similarity (raw A vs B):     {np.nanmean(angle_raw_list):.3f}")
    else:
        print("Avg joint-angle similarity (raw A vs B):     n/a")
    if angle_smooth_list:
        print(f"Avg joint-angle similarity (smooth A vs B):  {np.nanmean(angle_smooth_list):.3f}")
    else:
        print("Avg joint-angle similarity (smooth A vs B):  n/a")
    if embed_raw_list:
        print(f"Avg embedding similarity (raw A vs B):       {np.nanmean(embed_raw_list):.3f}")
    else:
        print("Avg embedding similarity (raw A vs B):       n/a")
    if embed_smooth_list:
        print(f"Avg embedding similarity (smooth A vs B):    {np.nanmean(embed_smooth_list):.3f}")
    else:
        print("Avg embedding similarity (smooth A vs B):    n/a")
    if worst_angle_raw[1] is not None:
        print(f"Worst joint-angle similarity (raw) at frame {worst_angle_raw[1]}:  {worst_angle_raw[0]:.3f}")
    if worst_angle_smooth[1] is not None:
        print(f"Worst joint-angle similarity (smooth) at frame {worst_angle_smooth[1]}: {worst_angle_smooth[0]:.3f}")

    # Plot worst frames for quick inspection
    if worst_raw_frames or worst_smooth_frames:
        plt.figure(figsize=(10, 8))
        rows = 0
        if worst_raw_frames:
            rows += 1
        if worst_smooth_frames:
            rows += 1
        row_idx = 1
        if worst_raw_frames:
            idx, score, fa, fb = worst_raw_frames
            plt.subplot(rows, 2, 1)
            plt.title(f"Raw worst frame {idx} (angle {score:.3f}) A")
            plt.axis('off')
            plt.imshow(fa)
            plt.subplot(rows, 2, 2)
            plt.title(f"Raw worst frame {idx} (angle {score:.3f}) B")
            plt.axis('off')
            plt.imshow(fb)
            row_idx += 1
        if worst_smooth_frames:
            idx, score, fa, fb = worst_smooth_frames
            plt.subplot(rows, 2, (row_idx - 1) * 2 + 1)
            plt.title(f"Smooth worst frame {idx} (angle {score:.3f}) A")
            plt.axis('off')
            plt.imshow(fa)
            plt.subplot(rows, 2, (row_idx - 1) * 2 + 2)
            plt.title(f"Smooth worst frame {idx} (angle {score:.3f}) B")
            plt.axis('off')
            plt.imshow(fb)
        plt.tight_layout()
        plt.show()

    if worst_jumps:
        print("\nTop joint jump outliers (disp px, frame, stream, joint, mean, std):")
        for d, fidx, stream, joint, mean, std, _ in sorted(worst_jumps, key=lambda x: x[0], reverse=True):
            print(f"  {d:.1f}px at frame {fidx} stream {stream} joint {joint} (mean {mean:.2f}, std {std:.2f})")

        # Plot worst jump frames (up to 3)
        plt.figure(figsize=(8, 4 * min(len(worst_jumps), 3)))
        for idx, (d, fidx, stream, joint, mean, std, frame_vis) in enumerate(sorted(worst_jumps, key=lambda x: x[0], reverse=True)[:3], start=1):
            plt.subplot(min(len(worst_jumps), 3), 1, idx)
            plt.title(f"Jump {d:.1f}px frame {fidx} stream {stream} joint {joint}\n(mean {mean:.2f}, std {std:.2f})")
            plt.axis('off')
            plt.imshow(frame_vis)
        plt.tight_layout()
        plt.show()
