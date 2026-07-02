from __future__ import annotations


import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from publication_suite.common import ensure_dir, repo_root


def read_csv_rows(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def find_first(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def save_fig(base: Path):
    base.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(base.with_suffix(".png"), bbox_inches="tight")
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def plot_learning_curve(out_dir: Path):
    episode_file = find_first(
        [
            out_dir / "matrix_episodes.csv",
            out_dir / "ablation_episodes.csv",
            out_dir / "robustness_episodes.csv",
            out_dir / "scalability_episodes.csv",
        ]
    )
    if episode_file is None:
        return
    rows = read_csv_rows(episode_file)
    if not rows:
        return
    preferred = [
        ("td3", "learned", "enhanced", "flight_aware", "imperfect"),
        ("td3", "learned", "enhanced", "flight_aware", "perfect"),
        ("td3", "fixed", "legacy", "legacy", "perfect"),
    ]
    selected = rows
    for algo, tr, obs, rew, csi in preferred:
        cand = [
            r for r in rows
            if r.get("algorithm", "") == algo
            and r.get("trajectory_mode", "") == tr
            and r.get("observation_mode", "") == obs
            and r.get("reward_mode", "") == rew
            and r.get("csi_mode", "") == csi
        ]
        if cand:
            selected = cand
            break
    by_ep = defaultdict(list)
    for r in selected:
        by_ep[int(float(r["episode"]))].append(float(r["episode_reward"]))
    xs = sorted(by_ep)
    ys = [np.mean(by_ep[x]) for x in xs]
    yerr = [np.std(by_ep[x]) if len(by_ep[x]) > 1 else 0.0 for x in xs]
    plt.figure()
    plt.errorbar(xs, ys, yerr=yerr, marker="o", linestyle="-")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    save_fig(out_dir / "learning_curve")


def plot_bar_from_summary(csv_path: Path, x_key: str, y_key: str, title: str, out_path: Path, hue_key: str | None = None):
    if not csv_path.exists():
        return
    rows = read_csv_rows(csv_path)
    if not rows:
        return
    plt.figure(figsize=(8, 4))
    if hue_key:
        groups = defaultdict(list)
        for r in rows:
            groups[r[hue_key]].append(r)
        for label, group_rows in groups.items():
            xs = [r[x_key] for r in group_rows]
            ys = [float(r[y_key]) for r in group_rows]
            plt.plot(xs, ys, marker="o", label=label)
        plt.legend()
    else:
        xs = [r[x_key] for r in rows]
        ys = [float(r[y_key]) for r in rows]
        plt.bar(xs, ys)
    plt.title(title)
    plt.xlabel(x_key.replace("_", " ").title())
    plt.ylabel(y_key.replace("_", " ").title())
    save_fig(out_path)


def main():
    p = argparse.ArgumentParser(description="Generate publication-ready figures from benchmark outputs.")
    p.add_argument("--input-dir", default="publication_suite_outputs")
    args = p.parse_args()
    root = repo_root()
    in_dir = root / args.input_dir

    plot_learning_curve(in_dir)

    plot_bar_from_summary(
        in_dir / "ablation" / "ablation_summary.csv",
        x_key="method",
        y_key="episode_reward",
        title="Ablation: Harvested Energy",
        out_path=in_dir / "ablation_energy_harvesting",
    )
    plot_bar_from_summary(
        in_dir / "ablation" / "ablation_summary.csv",
        x_key="method",
        y_key="energy_efficiency",
        title="Ablation: Energy Efficiency",
        out_path=in_dir / "ablation_energy_efficiency",
    )
    plot_bar_from_summary(
        in_dir / "ablation" / "ablation_summary.csv",
        x_key="method",
        y_key="flight_energy",
        title="Ablation: Flight Energy Consumption",
        out_path=in_dir / "ablation_flight_energy",
    )
    plot_bar_from_summary(
        in_dir / "matrix" / "matrix_summary.csv",
        x_key="csi_mode",
        y_key="energy_efficiency",
        title="Perfect CSI vs Imperfect CSI",
        out_path=in_dir / "csi_comparison",
    )
    plot_bar_from_summary(
        in_dir / "scalability" / "scalability_summary.csv",
        x_key="user_count",
        y_key="energy_efficiency",
        title="Scalability vs Number of Users",
        out_path=in_dir / "scalability_energy_efficiency",
    )


if __name__ == "__main__":
    main()
