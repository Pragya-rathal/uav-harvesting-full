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
    build_scalability_user_counts,
    ensure_dir,
    mean_std_ci,
    flatten_episode_metrics,
    repo_root,
    run_worker_subprocess,
    write_csv,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-user scalability evaluation.")
    p.add_argument("--output-dir", default="publication_suite_outputs/scalability")
    p.add_argument("--manifest", default=None)
    p.add_argument("--algorithm", default="td3", choices=["td3", "ddpg", "sd3"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--trajectory-mode", default="learned", choices=["fixed", "learned"])
    p.add_argument("--observation-mode", default="enhanced", choices=["legacy", "enhanced"])
    p.add_argument("--reward-mode", default="flight_aware", choices=["legacy", "flight_aware"])
    p.add_argument("--csi-mode", default="imperfect", choices=["perfect", "imperfect"])
    p.add_argument("--trajectory-source", default="Kmeans", choices=["Kmeans", "Fermat"])
    p.add_argument("--sigma-csi", type=float, default=0.05)
    p.add_argument("--flight-lambda", type=float, default=1e-3)
    p.add_argument("--prepare-data", action="store_true", default=True)
    p.add_argument("--no-prepare-data", dest="prepare_data", action="store_false")
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

    if args.prepare_data:
        gen_script = root / "publication_suite" / "generate_user_trajectories.py"
        # Generate 5-user support files if they are missing; harmless if already present.
        import subprocess, sys
        subprocess.run([sys.executable, str(gen_script), "--counts", "1", "3", "5"], check=True)

    user_counts = build_scalability_user_counts()
    raw_rows: List[Dict[str, Any]] = []
    episodes_rows: List[Dict[str, Any]] = []
    for user_count in user_counts:
        algo_spec = manifest[args.algorithm]
        scenario_root = algo_spec["scenario_root"]
        if user_count == 1:
            # preserve single-user baseline by using the single-user scenario root
            scenario_root = str((root / f"{args.algorithm.upper()}-SingleUT-Time").resolve()) if "SingleUT" not in scenario_root else scenario_root
        elif user_count == 3:
            scenario_root = str((root / f"{args.algorithm.upper()}-MultiUT-Time").resolve()) if "MultiUT" not in scenario_root else scenario_root
        elif user_count == 5:
            scenario_root = str((root / f"{args.algorithm.upper()}-MultiUT-Time").resolve()) if "MultiUT" not in scenario_root else scenario_root

        model_path = algo_spec["model_path"]
        for seed in args.seeds:
            payload = {
                "scenario_root": scenario_root,
                "algo": args.algorithm,
                "model_path": model_path,
                "trajectory_mode": args.trajectory_mode,
                "observation_mode": args.observation_mode,
                "reward_mode": args.reward_mode,
                "csi_mode": args.csi_mode,
                "trajectory_source": args.trajectory_source,
                "sigma_csi": args.sigma_csi,
                "flight_lambda": args.flight_lambda,
                "num_users": user_count if user_count > 1 else None,
                "seed": seed,
                "episodes": args.episodes,
            }
            result = run_worker_subprocess(worker, payload, check=True)
            raw_rows.append(
                {
                    "algorithm": args.algorithm,
                    "user_count": user_count,
                    "seed": seed,
                    "episode_reward": result["summary"]["episode_reward"],
                    "received_energy": result["summary"]["received_energy"],
                    "energy_efficiency": result["summary"]["energy_efficiency"],
                    "flight_energy": result["summary"]["flight_energy"],
                    "trajectory_length": result["summary"]["trajectory_length"],
                    "runtime_sec": result["runtime_sec"],
                }
            )

    write_csv(out_dir / "scalability_runs.csv", raw_rows)
    write_json(out_dir / "scalability_runs.json", raw_rows)
    write_csv(out_dir / "scalability_episodes.csv", episodes_rows)
    write_json(out_dir / "scalability_episodes.json", episodes_rows)

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault(int(row["user_count"]), []).append(row)
    summary_rows: List[Dict[str, Any]] = []
    for user_count in sorted(grouped):
        rows = grouped[user_count]
        summary_rows.append(
            {
                "user_count": user_count,
                "episode_reward": mean_std_ci([r["episode_reward"] for r in rows]),
                "received_energy": mean_std_ci([r["received_energy"] for r in rows]),
                "energy_efficiency": mean_std_ci([r["energy_efficiency"] for r in rows]),
                "flight_energy": mean_std_ci([r["flight_energy"] for r in rows]),
                "trajectory_length": mean_std_ci([r["trajectory_length"] for r in rows]),
                "runtime_sec": mean_std_ci([r["runtime_sec"] for r in rows]),
            }
        )

    write_json(out_dir / "scalability_summary.json", summary_rows)
    write_csv(
        out_dir / "scalability_summary.csv",
        [
            {
                "user_count": r["user_count"],
                "episode_reward": r["episode_reward"]["mean"],
                "episode_reward_std": r["episode_reward"]["std"],
                "received_energy": r["received_energy"]["mean"],
                "received_energy_std": r["received_energy"]["std"],
                "energy_efficiency": r["energy_efficiency"]["mean"],
                "energy_efficiency_std": r["energy_efficiency"]["std"],
                "flight_energy": r["flight_energy"]["mean"],
                "flight_energy_std": r["flight_energy"]["std"],
                "trajectory_length": r["trajectory_length"]["mean"],
                "trajectory_length_std": r["trajectory_length"]["std"],
                "runtime_sec": r["runtime_sec"]["mean"],
                "runtime_sec_std": r["runtime_sec"]["std"],
            }
            for r in summary_rows
        ],
    )


if __name__ == "__main__":
    main()
