import itertools
import numpy as np
from tqdm import tqdm

from src.model.kalman import create_classic_kalman, create_kalman_with_acceleration
from src.pose_pipeline import run_dual_pose_pipeline


def make_factory(base_fn, q_pos, q_vel, r_xy, extra=None):
    extra = extra or {}

    def factory(x, y, dt=1.0):
        # classic kalman factory doesn't take dt; accel does
        if base_fn is create_classic_kalman:
            kf = base_fn(x, y)
        else:
            kf = base_fn(x, y, dt=dt)
        if kf.x.shape[0] == 4:
            kf.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
            kf.R = np.diag([r_xy, r_xy])
        else:
            q_acc = extra.get("q_acc", q_vel)
            r_vel = extra.get("r_vel", r_xy * 0.5)
            r_acc = extra.get("r_acc", r_xy * 0.25)
            kf.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_acc, q_acc])
            kf.R = np.diag([r_xy, r_xy, r_vel, r_vel, r_acc, r_acc])
        return kf

    return factory


def simple_score(angle_list, embed_list):
    if not angle_list or not embed_list:
        return -np.inf
    angle = np.array(angle_list)
    embed = np.array(embed_list)
    return float(np.nanmean(angle) + np.nanmean(embed) - 0.1 * np.nanstd(angle) - 0.1 * np.nanstd(embed))


def grid_search(video_a, video_b, out_a, out_b, target_fps=30, mode="classic"):
    if mode == "classic":
        # Small grid around current good params
        q_pos_grid = [2e-3, 5e-3]
        q_vel_grid = [1e-2, 2e-2]
        r_xy_grid = [1.0, 4.0, 7.0]
        combos = itertools.product(q_pos_grid, q_vel_grid, r_xy_grid)
        base_fn = create_classic_kalman
    else:
        # Hand-picked small set (~6 combos) around known good accel params
        accel_combos = [
            (2e-3, 1e-2, 8e-2, 1.0, 3.0, 6.0),
            (2e-3, 1e-2, 8e-2, 3.0, 3.0, 6.0),
            (2e-3, 2e-2, 8e-2, 1.0, 3.0, 6.0),
            (2e-3, 2e-2, 8e-2, 3.0, 3.0, 6.0),
            (5e-3, 1e-2, 8e-2, 1.0, 3.0, 6.0),
            (5e-3, 2e-2, 8e-2, 3.0, 3.0, 6.0),
        ]
        combos = accel_combos
        base_fn = create_kalman_with_acceleration

    best = (-np.inf, None)
    for combo in tqdm(list(combos), desc=f"grid-{mode}"):
        if mode == "classic":
            q_pos, q_vel, r_xy = combo
            factory = make_factory(base_fn, q_pos, q_vel, r_xy)
            label = {"q_pos": q_pos, "q_vel": q_vel, "r_xy": r_xy}
        else:
            q_pos, q_vel, q_acc, r_xy, r_vel, r_acc = combo
            factory = make_factory(base_fn, q_pos, q_vel, r_xy, extra={"q_acc": q_acc, "r_vel": r_vel, "r_acc": r_acc})
            label = {"q_pos": q_pos, "q_vel": q_vel, "q_acc": q_acc, "r_xy": r_xy, "r_vel": r_vel, "r_acc": r_acc}

        scores = run_dual_pose_pipeline(
            video_a,
            video_b,
            out_a,
            out_b,
            factory,
            canonical_lengths=None,
            live=False,
            target_fps=target_fps,
            eval_only=True,
            max_frames=180,
        )
        score = simple_score(scores["angle_raw"], scores["embed_raw"])
        if score > best[0]:
            best = (score, label)

    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_a", required=True)
    parser.add_argument("--video_b", required=True)
    parser.add_argument("--out_a", required=True)
    parser.add_argument("--out_b", required=True)
    parser.add_argument("--mode", choices=["classic", "accel"], default="classic")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    best = grid_search(args.video_a, args.video_b, args.out_a, args.out_b, target_fps=args.fps, mode=args.mode)
    print("Best:", best)
