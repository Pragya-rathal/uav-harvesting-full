from __future__ import annotations


import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from publication_suite.common import (
    build_default_manifest,
    ensure_dir,
    mean_std_ci,
    flatten_episode_metrics,
    repo_root,
    run_worker_subprocess,
    write_csv,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Seed-sweep robustness evaluation.")
    p.add_argument("--output-dir", default="publication_suite_outputs/robustness")
    p.add_argument("--manifest", default=None)
    p.add_argument("--algorithm", default="td3", choices=["td3", "ddpg", "sd3"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--trajectory-mode", default="learned", choices=["fixed", "learned"])
    p.add_argument("--observation-mode", default="enhanced", choices=["legacy", "enhanced"])
    p.add_argument("--reward-mode", default="flight_aware", choices=["legacy", "flight_aware"])
    p.add_argument("--csi-mode", default="imperfect", choices=["perfect", "imperfect"])
    p.add_argument("--trajectory-source", default="Kmeans", choices=["Kmeans", "Fermat"])
    p.add_argument("--sigma-csi", type=float, default=0.05)
    p.add_argument("--flight-lambda", type=float, default=1e-3)
    return p.parse_args()


def load_manifest(path: str | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return build_default_manifest()
    return json.loads(Path(path).read_text())


def main():
    args = parse_args()
    root = repo_root()
    out_dir = ensure_dir(root / args.output_dir)
    worker = root / "publication_suite" / "collect_metrics.py"
    manifest = load_manifest(args.manifest)
    spec = manifest[args.algorithm]

    raw_rows: List[Dict[str, Any]] = []
    episodes_rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        payload = {
            "scenario_root": spec["scenario_root"],
            "algo": args.algorithm,
            "model_path": spec["model_path"],
            "trajectory_mode": args.trajectory_mode,
            "observation_mode": args.observation_mode,
            "reward_mode": args.reward_mode,
            "csi_mode": args.csi_mode,
            "trajectory_source": args.trajectory_source,
            "sigma_csi": args.sigma_csi,
            "flight_lambda": args.flight_lambda,
            "seed": seed,
            "episodes": args.episodes,
        }
        result = run_worker_subprocess(worker, payload, check=True)
        raw_rows.append(
            {
                "seed": seed,
                "episode_reward": result["summary"]["episode_reward"],
                "received_energy": result["summary"]["received_energy"],
                "energy_efficiency": result["summary"]["energy_efficiency"],
                "flight_energy": result["summary"]["flight_energy"],
                "trajectory_length": result["summary"]["trajectory_length"],
                "runtime_sec": result["runtime_sec"],
            }
        )

    write_csv(out_dir / "robustness_runs.csv", raw_rows)
    write_json(out_dir / "robustness_runs.json", raw_rows)
    write_csv(out_dir / "robustness_episodes.csv", episodes_rows)
    write_json(out_dir / "robustness_episodes.json", episodes_rows)
    summary = {
        "algorithm": args.algorithm,
        "trajectory_mode": args.trajectory_mode,
        "observation_mode": args.observation_mode,
        "reward_mode": args.reward_mode,
        "csi_mode": args.csi_mode,
        "trajectory_source": args.trajectory_source,
        "episode_reward": mean_std_ci([r["episode_reward"] for r in raw_rows]),
        "received_energy": mean_std_ci([r["received_energy"] for r in raw_rows]),
        "energy_efficiency": mean_std_ci([r["energy_efficiency"] for r in raw_rows]),
        "flight_energy": mean_std_ci([r["flight_energy"] for r in raw_rows]),
        "trajectory_length": mean_std_ci([r["trajectory_length"] for r in raw_rows]),
        "runtime_sec": mean_std_ci([r["runtime_sec"] for r in raw_rows]),
    }
    write_json(out_dir / "robustness_summary.json", summary)
    write_csv(
        out_dir / "performance_summary.csv",
        [
            {
                "algorithm": args.algorithm,
                "metric": "episode_reward",
                **summary["episode_reward"],
            },
            {
                "algorithm": args.algorithm,
                "metric": "received_energy",
                **summary["received_energy"],
            },
            {
                "algorithm": args.algorithm,
                "metric": "energy_efficiency",
                **summary["energy_efficiency"],
            },
            {
                "algorithm": args.algorithm,
                "metric": "flight_energy",
                **summary["flight_energy"],
            },
            {
                "algorithm": args.algorithm,
                "metric": "trajectory_length",
                **summary["trajectory_length"],
            },
        ],
    )


if __name__ == "__main__":
    main()
