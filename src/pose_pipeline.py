import sys, os
import pickle
from pathlib import Path
sys.path.append(os.path.abspath(".."))
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
from collections import deque
from src.model.media_pipe_pose import MediaPipePose
from src.model.embedding_model import PoseEmbeddingNet
from src.util.visualizer import visualize_multiple, visualize_pose, LIMBS
from src.parser.live_video_parser import VideoParser
from src.util.evaluation import compute_joint_angles, pairwise_distance_error


def smooth_classic(kalman_filters, num_joints, pred_keypoints):
    smoothed_keypoints = np.zeros_like(pred_keypoints)
    for joint_idx in range(num_joints):
        kalman = kalman_filters[joint_idx]
        xy = pred_keypoints[joint_idx][:2]
        if np.any(np.isnan(xy)):
            continue
        control_dim = kalman.B.shape[1] if hasattr(kalman, "B") else 0
        u = np.zeros(control_dim) if control_dim > 0 else 0.0
        kalman.predict(u) 
        meas_dim = kalman.H.shape[0]
        z = np.zeros(meas_dim)
        z[0] = xy[0]
        z[1] = xy[1]
        xy_smoothed = kalman.update(z)      # measurement uses x,y with zeros for others
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


def smooth_flow_with_accel(
    kalman_filters,
    prev_keypoints,
    num_joints,
    pred_keypoints,
    conf_threshold=0.6,
    vel_state=None,
    fallback_counts=None,
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

        # If measurement is non-finite but we have a previous finite point, reuse it to avoid NaNs
        if not np.all(np.isfinite(xy)) and prev_xy is not None and np.all(np.isfinite(prev_xy)):
            xy = prev_xy
            conf = 0.0

        # If we have no history yet, accept any finite measurement even if conf is low.
        use_meas = np.all(np.isfinite(xy)) and np.isfinite(conf) and (conf >= conf_threshold or prev_keypoints is None)

        if use_meas:
            # Ignore flow for position; set velocities to zero
            meas = np.array([xy[0], xy[1], 0.0, 0.0, 0.0, 0.0])
            state_upd = kalman.update(meas)
            smoothed_keypoints[joint_idx, :2] = state_upd[:2]
            smoothed_keypoints[joint_idx, 2] = conf if np.isfinite(conf) else 0.0
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


# Simple match thresholds; adjust as needed for your data.
ANGLE_MATCH_THR = 0.9
EMBED_MATCH_THR = 0.9
JUMP_STD_FACTOR = 3.0
JUMP_MIN_PX = 10.0
# If a measurement jumps more than this many pixels from the previous smoothed point,
# treat it as unreliable and fall back to flow-based extrapolation (see MAX_DISP_PX above).

def normalize_keypoints(kp):
        kp = kp.copy()
        xy = kp[:, :2]
        # Normalize to unit box based on current pose spread (training used normalized coords)
        min_xy = np.nanmin(xy, axis=0)
        max_xy = np.nanmax(xy, axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)
        xy_norm = (xy - min_xy) / span
        kp[:, :2] = xy_norm
        return kp

def safe_cosine(a, b):
        if a is None or b is None:
            return None
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return None
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return None
        return float(np.dot(a, b) / (na * nb))

def compute_embed(pred_a, pred_b, embed_model):
    try:
        emb_sa = emb_sb = None
        with torch.no_grad():
            norm_sa = normalize_keypoints(pred_a)
            norm_sb = normalize_keypoints(pred_b)
            emb_sa = embed_model(torch.tensor(norm_sa, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            emb_sb = embed_model(torch.tensor(norm_sb, dtype=torch.float32).unsqueeze(0)).numpy()[0]

            embed_score = safe_cosine(emb_sa, emb_sb) if emb_sa is not None and emb_sb is not None else None

            return embed_score
    except Exception:
        return None


def compute_pair_metrics(pred_a, smooth_a, pred_b, smooth_b, embed_model):
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
            norm_pred_a = normalize_keypoints(pred_a)
            norm_pred_b = normalize_keypoints(pred_b)
            emb_a = embed_model(torch.tensor(norm_pred_a, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            emb_b = embed_model(torch.tensor(norm_pred_b, dtype=torch.float32).unsqueeze(0)).numpy()[0]
            emb_sa = emb_sb = None
            if np.all(np.isfinite(smooth_a)) and np.all(np.isfinite(smooth_b)):
                norm_sa = normalize_keypoints(smooth_a)
                norm_sb = normalize_keypoints(smooth_b)
                emb_sa = embed_model(torch.tensor(norm_sa, dtype=torch.float32).unsqueeze(0)).numpy()[0]
                emb_sb = embed_model(torch.tensor(norm_sb, dtype=torch.float32).unsqueeze(0)).numpy()[0]

        embed_raw = safe_cosine(emb_a, emb_b)
        embed_smooth = safe_cosine(emb_sa, emb_sb) if emb_sa is not None and emb_sb is not None else None

        return angle_raw, embed_raw, angle_smooth, embed_smooth
    except Exception:
        return None, None, None, None


def stabilize_scale_with_bbox(keypoints2d, bbox_state, momentum=0.7, min_size=20.0, max_scale=1.5):
    """
    Simple scale stabilizer: keep a running average bbox and gently scale toward it.
    """
    kp = keypoints2d.copy()
    xy = kp[:, :2]
    if not np.all(np.isfinite(xy)):
        return kp, bbox_state
    x1, y1 = np.min(xy, axis=0)
    x2, y2 = np.max(xy, axis=0)
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    if w < min_size and h < min_size:
        return kp, bbox_state
    cur_bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
    cur_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    cur_wh = np.array([w, h], dtype=np.float32)

    # Init EMA
    if bbox_state.get("ema") is None:
        bbox_state["ema"] = cur_bbox
        bbox_state["ema_wh"] = cur_wh
        return kp, bbox_state

    # Update EMA of bbox corners and size
    bbox_state["ema"] = momentum * bbox_state["ema"] + (1 - momentum) * cur_bbox
    bbox_state["ema_wh"] = momentum * bbox_state.get("ema_wh", cur_wh) + (1 - momentum) * cur_wh

    target_bbox = bbox_state["ema"]
    target_wh = bbox_state["ema_wh"]
    target_center = np.array(
        [(target_bbox[0] + target_bbox[2]) / 2.0, (target_bbox[1] + target_bbox[3]) / 2.0],
        dtype=np.float32,
    )

    # Uniform scale toward EMA size
    scale_x = target_wh[0] / max(cur_wh[0], 1e-6)
    scale_y = target_wh[1] / max(cur_wh[1], 1e-6)
    scale = np.clip(np.sqrt(scale_x * scale_y), 1.0 / max_scale, max_scale)

    xy_scaled = (xy - cur_center) * scale + target_center
    kp[:, :2] = xy_scaled
    return kp, bbox_state


def export_video_keypoints_to_pkl(
    video_path,
    out_pkl_path,
    create_kalman,
    canonical_lengths=None,
    target_fps=30,
    max_frames=None,
    use_smoothed=True,
    static_image_mode=False,
    model_complexity=0,
):
    """
    Process a single video and save per-frame pose keypoints in AIST++ keypoints2d format.
    Output structure matches AIST++: {'keypoints2d': [np.ndarray(num_frames, 17, 3)], 'metadata': {...}}.
    """
    dt = 1 / target_fps
    parser = VideoParser(video_path)
    width = int(parser.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(parser.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = float(parser.cap.get(cv2.CAP_PROP_FPS))

    model = MediaPipePose(static_image_mode=static_image_mode, model_complexity=model_complexity)

    state = {
        "kalman_filters": [],
        "prev_gray_small": None,
        "prev_flow": None,
        "prev_keypoints": None,
        "prev_prev_keypoints": None,
        "prev_raw": None,
        "prev_prev_raw": None,
        "prev_conf": None,
        "flow_history": deque(maxlen=5),
        "vel_state": None,
        "fallback_counts": None,
        "canonical_lengths": canonical_lengths if canonical_lengths is not None and len(canonical_lengths) == len(LIMBS) else None,
        "bbox_state": {"ema": None, "ema_wh": None, "prev": None},
    }

    frames_out = []
    frame_idx = 0
    for pkt in parser:
        frame = pkt["frame"]
        pred_kp, smooth_kp, state = process_frame(frame, state, model, create_kalman, dt)
        frames_out.append(smooth_kp if use_smoothed else pred_kp)
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    keypoints_array = np.stack(frames_out, axis=0).astype(np.float32)
    data = {
        "keypoints2d": [keypoints_array],
        "metadata": {
            "video_path": str(video_path),
            "num_frames": int(keypoints_array.shape[0]),
            "image_size": (int(height), int(width)),
            "source_fps": fps_in,
            "target_fps": target_fps,
            "use_smoothed": use_smoothed,
        },
    }

    out_dir = Path(out_pkl_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_pkl_path, "wb") as f:
        pickle.dump(data, f)

    return out_pkl_path, data

def process_frame(frame, state, model, create_kalman, dt):
        # Basic preprocessing: no segmentation, no flow weighting
        gray_full = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        scale = 0.5
        gray_small = cv2.resize(gray_full, (0, 0), fx=scale, fy=scale)
        flow = np.zeros((gray_full.shape[0], gray_full.shape[1], 2), dtype=np.float32)

        # MediaPipe expects RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kp_img33 = model.detect_landmarks(frame_rgb)

        kp_img = model.convert_to_aist17(kp_img33)

        # Raw keypoints (used for visualization/metrics)
        pred_keypoints2d_raw = np.stack([kp_img[:, 0] * frame.shape[1],
                                         kp_img[:, 1] * frame.shape[0],
                                         kp_img[:, 3]], axis=1)
        valid_mask = (
            np.isfinite(pred_keypoints2d_raw[:, 0])
            & np.isfinite(pred_keypoints2d_raw[:, 1])
            & np.isfinite(pred_keypoints2d_raw[:, 2])
            & (pred_keypoints2d_raw[:, 2] > 0)
        )
        if valid_mask.sum() == 0:
            return pred_keypoints2d_raw, pred_keypoints2d_raw, state
        # Stabilized copy used for smoothing only
        pred_keypoints2d_stab, state["bbox_state"] = stabilize_scale_with_bbox(
            pred_keypoints2d_raw, state["bbox_state"]
        )

        def reinit_kf(jidx, xy_seed):
            x_seed, y_seed = xy_seed
            if not (np.isfinite(x_seed) and np.isfinite(y_seed)):
                x_seed = y_seed = 0.0
            try:
                kf = create_kalman(x_seed, y_seed, dt=dt)
            except TypeError:
                kf = create_kalman(x_seed, y_seed)
            state["kalman_filters"][jidx] = kf

        # No optical-flow-based override; rely on Kalman smoothing only

        if not state["kalman_filters"]:
            for joint_idx in range(pred_keypoints2d_raw.shape[0]):
                x_init, y_init = pred_keypoints2d_raw[joint_idx][:2]
                try:
                    kf = create_kalman(x_init, y_init, dt=dt)
                except TypeError:
                    kf = create_kalman(x_init, y_init)
                state["kalman_filters"].append(kf)

        # Use the smoother appropriate for the filter dimensionality
        if state["kalman_filters"] and state["kalman_filters"][0].x.shape[0] == 6:
            smoothed, state["vel_state"], state["fallback_counts"] = smooth_flow_with_accel(
                state["kalman_filters"],
                state["prev_keypoints"],
                pred_keypoints2d_stab.shape[0],
                pred_keypoints2d_stab,
                flow,
                state["prev_flow"],
                dt,
                conf_threshold=0.1,
                vel_state=state["vel_state"],
                prev_prev_keypoints=state["prev_prev_keypoints"],
                prev_raw=state["prev_raw"],
                prev_prev_raw=state["prev_prev_raw"],
                fallback_counts=state["fallback_counts"],
            )
        else:
            smoothed = smooth_classic(state["kalman_filters"], pred_keypoints2d_stab.shape[0], pred_keypoints2d_stab)
            state["vel_state"] = None
            state["fallback_counts"] = None

        # If any joint became NaN after smoothing but the measurement is finite, reseed that Kalman and use the measurement.
        for j in range(smoothed.shape[0]):
            xy_meas = pred_keypoints2d_stab[j, :2]
            if np.all(np.isfinite(smoothed[j, :2])):
                continue
            if np.all(np.isfinite(xy_meas)):
                reinit_kf(j, xy_meas)
                smoothed[j, :2] = xy_meas
                smoothed[j, 2] = pred_keypoints2d_stab[j, 2]

        state["prev_flow"] = flow
        state["prev_gray_small"] = gray_small
        state["prev_prev_keypoints"] = state["prev_keypoints"]
        state["prev_keypoints"] = smoothed
        state["prev_prev_raw"] = state["prev_raw"]
        state["prev_raw"] = pred_keypoints2d_stab

        return pred_keypoints2d_raw, smoothed, state

def run_dual_pose_pipeline(video_path_a, video_path_b, out_path_a, out_path_b, create_kalman, canonical_lengths=None, live=False, target_fps=30,
                           eval_only=False, max_frames=None,
                           raw_offset_frames=0, out_path_raw_aligned_a=None, out_path_raw_aligned_b=None,
                           smooth_offset_frames=0, out_path_smooth_aligned_a=None, out_path_smooth_aligned_b=None):
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = target_fps
    dt = 1 / fps

    parser_a = VideoParser(video_path_a)
    parser_b = VideoParser(video_path_b)

    w_a = int(parser_a.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_a = int(parser_a.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_b = int(parser_b.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_b = int(parser_b.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Always process all frames; writers use target_fps
    fps_a_in = 60
    fps_b_in = 60

    writer_a = None if eval_only else cv2.VideoWriter(out_path_a, fourcc, fps, (w_a, h_a))
    writer_b = None if eval_only else cv2.VideoWriter(out_path_b, fourcc, fps, (w_b, h_b))
    writer_raw_a = writer_raw_b = None
    writer_smooth_a = writer_smooth_b = None
    if not eval_only and raw_offset_frames > 0:
        if out_path_raw_aligned_a:
            writer_raw_a = cv2.VideoWriter(out_path_raw_aligned_a, fourcc, fps, (w_a, h_a))
        if out_path_raw_aligned_b:
            writer_raw_b = cv2.VideoWriter(out_path_raw_aligned_b, fourcc, fps, (w_b, h_b))
    if not eval_only and smooth_offset_frames > 0:
        if out_path_smooth_aligned_a:
            writer_smooth_a = cv2.VideoWriter(out_path_smooth_aligned_a, fourcc, fps, (w_a, h_a))
        if out_path_smooth_aligned_b:
            writer_smooth_b = cv2.VideoWriter(out_path_smooth_aligned_b, fourcc, fps, (w_b, h_b))

    model = MediaPipePose(static_image_mode=False, model_complexity=0)
    ckpt_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
    embed_path = os.path.join(ckpt_dir, "best_model.pth")
    embed_model = PoseEmbeddingNet(embed_dims=128, hidden_dims=[512, 512], dropout=0.3)
    embed_model.load_state_dict(torch.load(embed_path, map_location='cpu'))
    embed_model.eval()
    def init_state():
        return {
            "kalman_filters": [],
            "prev_gray_small": None,
            "prev_flow": None,
            "prev_keypoints": None,
            "prev_prev_keypoints": None,
            "prev_raw": None,
            "prev_prev_raw": None,
            "prev_conf": None,
            "flow_history": deque(maxlen=5),
            "vel_state": None,
            "fallback_counts": None,
            "canonical_lengths": canonical_lengths if canonical_lengths is not None and len(canonical_lengths) == len(LIMBS) else None,
            "bbox_state": {"ema": None, "ema_wh": None, "prev": None},
        }

    state_a = init_state()
    state_b = init_state()

    angle_raw_list = []
    angle_smooth_list = []
    embed_raw_list = []
    embed_smooth_list = []
    pdist_raw_list = []
    pdist_smooth_list = []
    worst_raw_frames = None  # (frame_idx, score, frame_a_vis, frame_b_vis)
    worst_smooth_frames = None
    worst_pdist_raw = (-np.inf, None)
    worst_pdist_smooth = (-np.inf, None)
    worst_embed_raw = (np.inf, None)
    worst_embed_smooth = (np.inf, None)
    top_angle_mins = []
    top_embed_mins = []
    top_angle_frames = []  # collect all frames to pick lowest later
    top_embed_frames = []
    min_embed_raw = (np.inf, None, None, None)  # (value, frame_idx, frame_a_vis, frame_b_vis)

    start_time = time.time()
    frame_idx = 0
    raw_queue_a = deque(maxlen=raw_offset_frames + 1) if raw_offset_frames > 0 else None
    raw_queue_b = deque(maxlen=raw_offset_frames + 1) if raw_offset_frames > 0 else None
    frame_queue_a = deque(maxlen=smooth_offset_frames + 1) if smooth_offset_frames > 0 else None
    frame_queue_b = deque(maxlen=smooth_offset_frames + 1) if smooth_offset_frames > 0 else None


    for pkt_idx, (pkt_a, pkt_b) in enumerate(zip(parser_a, parser_b)):
        frame_a = pkt_a['frame']
        frame_b = pkt_b['frame']

        pred_a, smooth_a, state_a = process_frame(frame_a, state_a, model, create_kalman, dt)
        pred_b, smooth_b, state_b = process_frame(frame_b, state_b, model, create_kalman, dt)

        angle_raw, embed_raw, angle_smooth, embed_smooth = compute_pair_metrics(pred_a, smooth_a, pred_b, smooth_b, embed_model)
        pdist_raw = pairwise_distance_error(pred_a, pred_b)
        pdist_smooth = pairwise_distance_error(smooth_a, smooth_b)
        raw_is_worst = smooth_is_worst = False
        if angle_raw is not None:
            angle_raw_list.append(angle_raw)
            # track top local minima for angle_raw
            if len(angle_raw_list) >= 3:
                prev_val = angle_raw_list[-2]
                prev_prev = angle_raw_list[-3]
                if prev_val < angle_raw_list[-1] and prev_val < prev_prev:
                    top_angle_mins.append((len(angle_raw_list)-2, prev_val))
        if angle_smooth is not None:
            angle_smooth_list.append(angle_smooth)
        if embed_raw is not None:
            embed_raw_list.append(embed_raw)
            if len(embed_raw_list) >= 3:
                prev_val = embed_raw_list[-2]
                prev_prev = embed_raw_list[-3]
                if prev_val < embed_raw_list[-1] and prev_val < prev_prev:
                    top_embed_mins.append((len(embed_raw_list)-2, prev_val))
            if embed_raw < worst_embed_raw[0]:
                worst_embed_raw = (embed_raw, frame_idx)
        if embed_smooth is not None:
            embed_smooth_list.append(embed_smooth)
            if embed_smooth < worst_embed_smooth[0]:
                worst_embed_smooth = (embed_smooth, frame_idx)
        if pdist_raw is not None:
            pdist_raw_list.append(pdist_raw)
            if pdist_raw > worst_pdist_raw[0]:
                worst_pdist_raw = (pdist_raw, frame_idx)
                raw_is_worst = True
        if pdist_smooth is not None:
            pdist_smooth_list.append(pdist_smooth)
            if pdist_smooth > worst_pdist_smooth[0]:
                worst_pdist_smooth = (pdist_smooth, frame_idx)
                smooth_is_worst = True

        elapsed = time.time() - start_time
        current_fps = frame_idx / elapsed if elapsed > 0 else 0

        frame_a_vis = visualize_multiple(frame_a.copy(), pred_a, smooth_a, fps=current_fps)
        frame_b_vis = visualize_multiple(frame_b.copy(), pred_b, smooth_b, fps=current_fps)

        # Overlay metrics (raw) on each frame
        if pdist_raw is not None:
            cv2.putText(frame_a_vis, f"P-dist raw: {pdist_raw:.2f}", (10, frame_a_vis.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame_b_vis, f"P-dist raw: {pdist_raw:.2f}", (10, frame_b_vis.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        if angle_raw is not None:
            cv2.putText(frame_a_vis, f"Angle raw: {angle_raw:.2f}", (10, frame_a_vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame_b_vis, f"Angle raw: {angle_raw:.2f}", (10, frame_b_vis.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Collect frames for later display of worst local minima
        if angle_raw is not None:
            top_angle_frames.append((frame_idx, angle_raw, frame_a_vis.copy(), frame_b_vis.copy()))
        if embed_raw is not None:
            top_embed_frames.append((frame_idx, embed_raw, frame_a_vis.copy(), frame_b_vis.copy()))

        if writer_a is not None:
            writer_a.write(frame_a_vis)
        if writer_b is not None:
            writer_b.write(frame_b_vis)
        if raw_queue_a is not None:
            raw_queue_a.append(visualize_pose(frame_a.copy(), pred_a, color=(255, 0, 0)))
            if writer_raw_a is not None and len(raw_queue_a) > raw_offset_frames:
                writer_raw_a.write(raw_queue_a[0])
        if raw_queue_b is not None:
            raw_queue_b.append(visualize_pose(frame_b.copy(), pred_b, color=(255, 0, 0)))
            if writer_raw_b is not None and len(raw_queue_b) > raw_offset_frames:
                writer_raw_b.write(raw_queue_b[0])
        if frame_queue_a is not None:
            frame_queue_a.append(frame_a.copy())
            if writer_smooth_a is not None and len(frame_queue_a) > smooth_offset_frames:
                target_frame = frame_queue_a[0].copy()
                vis = visualize_pose(cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB), smooth_a, color=(0, 255, 0))
                writer_smooth_a.write(vis)
        if frame_queue_b is not None:
            frame_queue_b.append(frame_b.copy())
            if writer_smooth_b is not None and len(frame_queue_b) > smooth_offset_frames:
                target_frame = frame_queue_b[0].copy()
                vis = visualize_pose(cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB), smooth_b, color=(0, 255, 0))
                writer_smooth_b.write(vis)

        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    if writer_a is not None:
        writer_a.release()
    if writer_b is not None:
        writer_b.release()
    if writer_raw_a is not None:
        writer_raw_a.release()
    if writer_raw_b is not None:
        writer_raw_b.release()
    if writer_smooth_a is not None:
        writer_smooth_a.release()
    if writer_smooth_b is not None:
        writer_smooth_b.release()

    total_time = time.time() - start_time
    final_fps = frame_idx / total_time if total_time > 0 else 0

    if eval_only:
        angle_raw_min = np.nanmin(angle_raw_list) if angle_raw_list else np.nan
        angle_smooth_min = np.nanmin(angle_smooth_list) if angle_smooth_list else np.nan
        embed_raw_min = np.nanmin(embed_raw_list) if embed_raw_list else np.nan
        embed_smooth_min = np.nanmin(embed_smooth_list) if embed_smooth_list else np.nan
        pdist_raw_worst = worst_pdist_raw[0] if worst_pdist_raw[1] is not None else np.nan
        pdist_smooth_worst = worst_pdist_smooth[0] if worst_pdist_smooth[1] is not None else np.nan
        return {
            "angle_raw": angle_raw_list,
            "angle_smooth": angle_smooth_list,
            "embed_raw": embed_raw_list,
            "embed_smooth": embed_smooth_list,
            "pdist_raw": pdist_raw_list,
            "pdist_smooth": pdist_smooth_list,
            "angle_raw_min": angle_raw_min,
            "angle_smooth_min": angle_smooth_min,
            "embed_raw_min": embed_raw_min,
            "embed_smooth_min": embed_smooth_min,
            "pdist_raw_worst": pdist_raw_worst,
            "pdist_smooth_worst": pdist_smooth_worst,
        }

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
    if worst_embed_raw[1] is not None:
        print(f"Worst embedding similarity (raw) at frame {worst_embed_raw[1]}:  {worst_embed_raw[0]:.3f}")
    if worst_embed_smooth[1] is not None:
        print(f"Worst embedding similarity (smooth) at frame {worst_embed_smooth[1]}: {worst_embed_smooth[0]:.3f}")
    if pdist_raw_list:
        print(f"Avg pairwise-distance error (raw):           {np.nanmean(pdist_raw_list):.2f}")
    else:
        print("Avg pairwise-distance error (raw):           n/a")
    if pdist_smooth_list:
        print(f"Avg pairwise-distance error (smooth):        {np.nanmean(pdist_smooth_list):.2f}")
    else:
        print("Avg pairwise-distance error (smooth):        n/a")
    if worst_pdist_raw[1] is not None:
        print(f"Worst pairwise distance error (raw) at frame {worst_pdist_raw[1]}:  {worst_pdist_raw[0]:.3f}")
    if worst_pdist_smooth[1] is not None:
        print(f"Worst pairwise distance error (smooth) at frame {worst_pdist_smooth[1]}: {worst_pdist_smooth[0]:.3f}")

    # Plot pairwise distance error over time
    if pdist_raw_list:
        plt.figure(figsize=(10, 4))
        frames = np.arange(len(pdist_raw_list))
        plt.plot(frames, pdist_raw_list, label='Pairwise dist error (raw)')
        if pdist_smooth_list and len(pdist_smooth_list) == len(pdist_raw_list):
            plt.plot(frames, pdist_smooth_list, label='Pairwise dist error (smooth)')
        plt.xlabel('Frame')
        plt.ylabel('Error')
        plt.title('Pairwise distance error over time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    if embed_raw_list:
        plt.figure(figsize=(10, 4))
        frames = np.arange(len(embed_raw_list))
        plt.plot(frames, embed_raw_list, label='Embed raw')
        if top_embed_mins:
            mins_sorted = sorted(top_embed_mins, key=lambda x: x[1])[:7]
            for idx, val in mins_sorted:
                plt.axvline(idx, color='r', alpha=0.3)
                plt.scatter(idx, val, color='r')
        plt.xlabel('Frame')
        plt.ylabel('Similarity')
        plt.title('Embedding similarity over time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Show top-k lowest raw similarity frames (embed)
    if top_embed_frames:
        top_embed_frames_sorted = sorted(top_embed_frames, key=lambda x: x[1])[:7]
        rows = len(top_embed_frames_sorted)
        plt.figure(figsize=(10, 3 * rows))
        for i, (idx, val, fa, fb) in enumerate(top_embed_frames_sorted, start=1):
            plt.subplot(rows, 2, 2*i - 1)
            plt.title(f"Embed min frame {idx} ({val:.3f}) A")
            plt.axis('off')
            plt.imshow(cv2.cvtColor(fa, cv2.COLOR_BGR2RGB))
            plt.subplot(rows, 2, 2*i)
            plt.title(f"Embed min frame {idx} ({val:.3f}) B")
            plt.axis('off')
            plt.imshow(cv2.cvtColor(fb, cv2.COLOR_BGR2RGB))
        plt.tight_layout()
        plt.show()
    # Plot raw similarities over time
    frames = np.arange(len(angle_raw_list))
    if len(frames) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(frames, angle_raw_list, label='Angle raw')
        if embed_raw_list:
            plt.plot(frames[:len(embed_raw_list)], embed_raw_list, label='Embed raw')
        plt.xlabel('Frame')
        plt.ylabel('Similarity')
        plt.title('Similarity over time (raw)')
        plt.legend()
        plt.tight_layout()
        plt.show()
