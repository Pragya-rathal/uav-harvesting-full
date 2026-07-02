from __future__ import annotations


import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from publication_suite.common import (
    build_default_manifest,
    build_matrix_configs,
    ensure_dir,
    mean_std_ci,
    flatten_episode_metrics,
    repo_root,
    run_worker_subprocess,
    write_csv,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run the full 2x2x2x2 experiment matrix.")
    p.add_argument("--output-dir", default="publication_suite_outputs/matrix")
    p.add_argument("--manifest", default=None, help="Optional JSON manifest mapping algo->scenario/model paths.")
    p.add_argument("--algorithms", nargs="+", default=["td3", "ddpg", "sd3"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--trajectory-source", default="Kmeans", choices=["Kmeans", "Fermat"])
    p.add_argument("--sigma-csi", type=float, default=0.05)
    p.add_argument("--flight-lambda", type=float, default=1e-3)
    p.add_argument("--skip-missing", action="store_true", default=False)
    p.add_argument("--no-skip-missing", dest="skip_missing", action="store_false")
    return p.parse_args()


def load_manifest(path: str | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return build_default_manifest()
    payload = json.loads(Path(path).read_text())
    return payload


def main():
    args = parse_args()
    root = repo_root()
    out_dir = ensure_dir(root / args.output_dir)
    worker_script = root / "publication_suite" / "collect_metrics.py"
    manifest = load_manifest(args.manifest)

    raw_rows: List[Dict[str, Any]] = []
    episodes_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    configs = build_matrix_configs()

    for algo in args.algorithms:
        if algo not in manifest:
            skipped.append({"algorithm": algo, "reason": "missing_from_manifest"})
            continue
        spec = manifest[algo]
        scenario_root = spec["scenario_root"]
        model_path = spec["model_path"]
        for cfg in configs:
            for seed in args.seeds:
                payload = {
                    "scenario_root": scenario_root,
                    "algo": algo,
                    "model_path": model_path,
                    "trajectory_mode": cfg["trajectory_mode"],
                    "observation_mode": cfg["observation_mode"],
                    "reward_mode": cfg["reward_mode"],
                    "csi_mode": cfg["csi_mode"],
                    "trajectory_source": args.trajectory_source,
                    "sigma_csi": args.sigma_csi,
                    "flight_lambda": args.flight_lambda,
                    "seed": seed,
                    "episodes": args.episodes,
                }
                try:
                    result = run_worker_subprocess(worker_script, payload, check=True)
                    row = {
                        "algorithm": algo,
                        "scenario_root": scenario_root,
                        "model_path": model_path,
                        "trajectory_mode": cfg["trajectory_mode"],
                        "observation_mode": cfg["observation_mode"],
                        "reward_mode": cfg["reward_mode"],
                        "csi_mode": cfg["csi_mode"],
                        "trajectory_source": args.trajectory_source,
                        "seed": seed,
                        "episodes": args.episodes,
                        "status": "ok",
                        "episode_reward": result["summary"]["episode_reward"],
                        "received_energy": result["summary"]["received_energy"],
                        "energy_efficiency": result["summary"]["energy_efficiency"],
                        "flight_energy": result["summary"]["flight_energy"],
                        "trajectory_length": result["summary"]["trajectory_length"],
                        "normalized_episode_reward": result["summary"]["normalized_episode_reward"],
                        "runtime_sec": result["runtime_sec"],
                    }
                    raw_rows.append(row)
                    episodes_rows.extend(flatten_episode_metrics(row, result["episodes_data"]))
                except Exception as exc:
                    row = {
                        "algorithm": algo,
                        "scenario_root": scenario_root,
                        "model_path": model_path,
                        "trajectory_mode": cfg["trajectory_mode"],
                        "observation_mode": cfg["observation_mode"],
                        "reward_mode": cfg["reward_mode"],
                        "csi_mode": cfg["csi_mode"],
                        "trajectory_source": args.trajectory_source,
                        "seed": seed,
                        "episodes": args.episodes,
                        "status": "failed",
                        "error": str(exc),
                    }
                    raw_rows.append(row)
                    episodes_rows.extend(flatten_episode_metrics(row, result["episodes_data"]))
                    if not args.skip_missing:
                        raise

    write_csv(out_dir / "matrix_runs.csv", raw_rows)
    write_json(out_dir / "matrix_runs.json", raw_rows)
    write_csv(out_dir / "matrix_episodes.csv", episodes_rows)
    write_json(out_dir / "matrix_episodes.json", episodes_rows)

    # aggregate over successful runs
    grouped: Dict[tuple, List[Dict[str, Any]]] = {}
    for row in raw_rows:
        if row.get("status") != "ok":
            continue
        key = (
            row["algorithm"],
            row["trajectory_mode"],
            row["observation_mode"],
            row["reward_mode"],
            row["csi_mode"],
            row["trajectory_source"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        algorithm, trajectory_mode, observation_mode, reward_mode, csi_mode, trajectory_source = key
        summary_rows.append(
            {
                "algorithm": algorithm,
                "trajectory_mode": trajectory_mode,
                "observation_mode": observation_mode,
                "reward_mode": reward_mode,
                "csi_mode": csi_mode,
                "trajectory_source": trajectory_source,
                "n": len(rows),
                "episode_reward": mean_std_ci([r["episode_reward"] for r in rows]),
                "received_energy": mean_std_ci([r["received_energy"] for r in rows]),
                "energy_efficiency": mean_std_ci([r["energy_efficiency"] for r in rows]),
                "flight_energy": mean_std_ci([r["flight_energy"] for r in rows]),
                "trajectory_length": mean_std_ci([r["trajectory_length"] for r in rows]),
                "runtime_sec": mean_std_ci([r["runtime_sec"] for r in rows]),
            }
        )

    write_json(out_dir / "matrix_summary.json", summary_rows)
    write_csv(
        out_dir / "matrix_summary.csv",
        [
            {
                "algorithm": row["algorithm"],
                "trajectory_mode": row["trajectory_mode"],
                "observation_mode": row["observation_mode"],
                "reward_mode": row["reward_mode"],
                "csi_mode": row["csi_mode"],
                "trajectory_source": row["trajectory_source"],
                "n": row["n"],
                "episode_reward": row["episode_reward"]["mean"],
                "episode_reward_std": row["episode_reward"]["std"],
                "received_energy": row["received_energy"]["mean"],
                "received_energy_std": row["received_energy"]["std"],
                "energy_efficiency": row["energy_efficiency"]["mean"],
                "energy_efficiency_std": row["energy_efficiency"]["std"],
                "flight_energy": row["flight_energy"]["mean"],
                "flight_energy_std": row["flight_energy"]["std"],
                "trajectory_length": row["trajectory_length"]["mean"],
                "trajectory_length_std": row["trajectory_length"]["std"],
                "runtime_sec": row["runtime_sec"]["mean"],
                "runtime_sec_std": row["runtime_sec"]["std"],
            }
            for row in summary_rows
        ],
    )
    if skipped:
        write_json(out_dir / "skipped.json", skipped)


if __name__ == "__main__":
    main()
