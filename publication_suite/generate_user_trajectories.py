from __future__ import annotations


import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from publication_suite.common import ensure_dir, repo_root, write_json


def _move_toward(cur: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
    x, y = cur
    tx, ty = target
    if abs(tx - x) >= abs(ty - y):
        x = x - 1 if x > tx else x + 1 if x < tx else x
    else:
        y = y - 1 if y > ty else y + 1 if y < ty else y
    return x, y


def _trajectory(start: Tuple[int, int], target: Tuple[int, int]) -> np.ndarray:
    cur = start
    points = [(cur[0], cur[1], 1)]
    while cur != target:
        cur = _move_toward(cur, target)
        points.append((cur[0], cur[1], 1))
    return np.asarray(points, dtype=np.float32)


def _pad_to_length(arr: np.ndarray, length: int) -> np.ndarray:
    if len(arr) >= length:
        return arr
    pad = np.repeat(arr[-1][None, :], length - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0)


def _weizfeld(points: np.ndarray, eps: float = 1e-6, max_iter: int = 200) -> np.ndarray:
    x = np.mean(points, axis=0)
    for _ in range(max_iter):
        diffs = points - x[None, :]
        dists = np.linalg.norm(diffs, axis=1)
        if np.any(dists < eps):
            return points[np.argmin(dists)]
        weights = 1.0 / np.maximum(dists, eps)
        x_new = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
        if np.linalg.norm(x_new - x) < 1e-3:
            return x_new
        x = x_new
    return x


def _trajectory_from_points(points_per_t: Sequence[np.ndarray], mode: str) -> np.ndarray:
    out = []
    max_len = max(len(p) for p in points_per_t)
    padded = [_pad_to_length(p, max_len) for p in points_per_t]
    for t in range(max_len):
        pts = np.asarray([p[t, :2] for p in padded], dtype=np.float32)
        if mode.lower() == "fermat":
            xy = _weizfeld(pts)
        else:
            xy = np.mean(pts, axis=0)
        out.append([float(xy[0]), float(xy[1]), 20.0])
    return np.asarray(out, dtype=np.float32)


def _base_pairs(count: int, split: str):
    train_pairs = [
        ((0, 0), (20, 20)),
        ((20, 0), (0, 20)),
        ((20, 20), (0, 0)),
        ((0, 20), (20, 0)),
        ((10, 0), (10, 20)),
    ]
    test_pairs = [
        ((20, 0), (0, 20)),
        ((20, 20), (0, 0)),
        ((0, 20), (20, 0)),
        ((0, 0), (20, 20)),
        ((10, 20), (10, 0)),
    ]
    base = train_pairs if split == "Train" else test_pairs
    return [base[i % len(base)] for i in range(count)]


def generate_user_trajectories(count: int, split: str, out_dir: Path, overwrite: bool = False) -> List[Path]:
    out_dir = ensure_dir(out_dir)
    files: List[Path] = []
    pairs = _base_pairs(count, split)
    user_trajs = [_trajectory(s, t) for s, t in pairs]
    for idx, traj in enumerate(user_trajs):
        path = out_dir / f"{split}_Trajectory_UT_{idx}.csv"
        if overwrite or not path.exists():
            np.savetxt(path, traj, delimiter=",")
        files.append(path)

    fermat_path = out_dir / f"Fermat_{split}_Trajectory_{count}.csv"
    kmeans_path = out_dir / f"Kmeans_{split}_Trajectory_{count}.csv"
    if overwrite or not fermat_path.exists():
        np.savetxt(fermat_path, _trajectory_from_points(user_trajs, "Fermat"), delimiter=",")
    if overwrite or not kmeans_path.exists():
        np.savetxt(kmeans_path, _trajectory_from_points(user_trajs, "Kmeans"), delimiter=",")
    files.extend([fermat_path, kmeans_path])
    return files


def parse_args():
    p = argparse.ArgumentParser(description="Generate scalable user and UAV trajectory CSVs.")
    p.add_argument("--counts", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--splits", nargs="+", choices=["Train", "Test"], default=["Train", "Test"])
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    root = repo_root()
    out_dir = root / "CreateData"
    produced = []
    for count in args.counts:
        for split in args.splits:
            produced.extend([str(p.relative_to(root)) for p in generate_user_trajectories(count, split, out_dir, overwrite=args.overwrite)])
    write_json(root / "publication_suite_outputs" / "generated_trajectories.json", produced)
    print(f"Generated/verified {len(produced)} trajectory files")


if __name__ == "__main__":
    main()
