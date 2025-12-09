import os
import random
import glob
import warnings
import numpy as np
from collections import defaultdict
import absl.logging
from src.pose_pipeline import run_dual_pose_pipeline
from src.model.kalman import create_classic_kalman, create_kalman_with_acceleration


def parse_meta(path):
    """Parse genre, choreo (chXX), dancer (dXX), and m code from filename."""
    base = os.path.basename(path)
    parts = base.split('_')
    if len(parts) < 5:
        return None
    genre = parts[0]  # e.g., gBR
    ch = None
    dancer = None
    mcode = None
    for p in parts:
        if p.startswith('ch'):
            ch = p
        if p.startswith('d') and len(p) == 3:
            dancer = p
        if p.startswith('m') and len(p) >= 3:
            mcode = p
    return genre, ch, dancer, mcode


def sample_pairs(video_paths, n_matches=5, n_mismatches=5):
    meta = {}
    for p in video_paths:
        m = parse_meta(p)
        if m is None:
            continue
        meta[p] = m

    # Group by (genre, ch, mcode)
    buckets = defaultdict(list)
    for p, (genre, ch, dancer, mcode) in meta.items():
        if genre and ch and dancer and mcode:
            buckets[(genre, ch, mcode)].append(p)

    matches = []
    keys = list(buckets.keys())
    random.shuffle(keys)
    for k in keys:
        vids = buckets[k]
        if len(vids) < 2:
            continue
        pairs = []
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                pairs.append((vids[i], vids[j]))
        random.shuffle(pairs)
        for p in pairs:
            matches.append(p)
            if len(matches) >= n_matches:
                break
        if len(matches) >= n_matches:
            break

    # Mismatches: pick from different genres
    genres = defaultdict(list)
    for p, (genre, ch, dancer, mcode) in meta.items():
        genres[genre].append(p)
    genre_list = list(genres.keys())
    mismatches = []
    while len(mismatches) < n_mismatches and len(genre_list) >= 2:
        g1, g2 = random.sample(genre_list, 2)
        v1 = random.choice(genres[g1])
        v2 = random.choice(genres[g2])
        mismatches.append((v1, v2))
    return matches, mismatches


def eval_pairs(pairs, kalman_factory, tag, max_frames=180, target_fps=30):
    results = []
    for a, b in pairs:
        scores = run_dual_pose_pipeline(
            a, b,
            out_path_a='/tmp/a.mp4', out_path_b='/tmp/b.mp4',
            create_kalman=kalman_factory,
            canonical_lengths=None, live=False, target_fps=target_fps,
            eval_only=True, max_frames=max_frames,
        )
        results.append(scores)

    def agg_mean(key):
        vals = []
        for s in results:
            if key in s and s[key]:
                v = np.nanmean(s[key])
                if np.isfinite(v):
                    vals.append(v)
        return np.nanmean(vals) if vals else np.nan

    def agg_min(key):
        vals = []
        for s in results:
            if key in s and s[key]:
                v = np.nanmin(s[key])
                if np.isfinite(v):
                    vals.append(v)
        return np.nanmean(vals) if vals else np.nan

    def agg_max(key):
        vals = []
        for s in results:
            if key in s and s[key]:
                v = np.nanmax(s[key])
                if np.isfinite(v):
                    vals.append(v)
        return np.nanmean(vals) if vals else np.nan

    def agg_scalar(key):
        vals = []
        for s in results:
            if key in s:
                v = s[key]
                if np.isfinite(v):
                    vals.append(v)
        return np.nanmean(vals) if vals else np.nan

    return {
        'tag': tag,
        'embed_raw': agg_mean('embed_raw'),
        'embed_smooth': agg_mean('embed_smooth'),
        'pdist_raw': agg_mean('pdist_raw'),
        'pdist_smooth': agg_mean('pdist_smooth'),
        'embed_raw_min': agg_min('embed_raw'),
        'embed_smooth_min': agg_min('embed_smooth'),
        'pdist_raw_max': agg_max('pdist_raw'),
        'pdist_smooth_max': agg_max('pdist_smooth'),
        'embed_raw_min_pair': agg_scalar('embed_raw_min'),
        'embed_smooth_min_pair': agg_scalar('embed_smooth_min'),
        'pdist_raw_worst_pair': agg_scalar('pdist_raw_worst'),
        'pdist_smooth_worst_pair': agg_scalar('pdist_smooth_worst'),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/raw', help='Root of raw videos')
    parser.add_argument('--pattern', default='**/*.mp4', help='Glob pattern under root')
    parser.add_argument('--n_matches', type=int, default=5)
    parser.add_argument('--n_mismatches', type=int, default=5)
    parser.add_argument('--max_frames', type=int, default=180)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    all_videos = glob.glob(os.path.join(args.root, args.pattern), recursive=True)
    matches, mismatches = sample_pairs(all_videos, n_matches=args.n_matches, n_mismatches=args.n_mismatches)

    print(f"Found {len(matches)} matches and {len(mismatches)} mismatches")

    models = [
        ('classic', create_classic_kalman),
        ('accel', create_kalman_with_acceleration),
    ]

    for name, factory in models:
        m_res = eval_pairs(matches, factory, f'{name}-matches', max_frames=args.max_frames, target_fps=args.fps)
        mm_res = eval_pairs(mismatches, factory, f'{name}-mismatches', max_frames=args.max_frames, target_fps=args.fps)
        print(m_res)
        print(mm_res)


if __name__ == '__main__':
    main()
