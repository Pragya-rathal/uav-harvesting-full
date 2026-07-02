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
    build_td3_ablation_configs,
    ensure_dir,
    mean_std_ci,
    flatten_episode_metrics,
    repo_root,
    run_worker_subprocess,
    rows_to_latex_table,
    write_csv,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Run TD3 ablation study and export tables.")
    p.add_argument("--output-dir", default="publication_suite_outputs/ablation")
    p.add_argument("--manifest", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=5)
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
    spec = manifest["td3"]

    configs = build_td3_ablation_configs()
    raw_rows: List[Dict[str, Any]] = []
    episodes_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        for seed in args.seeds:
            payload = {
                "scenario_root": spec["scenario_root"],
                "algo": "td3",
                "model_path": spec["model_path"],
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
            result = run_worker_subprocess(worker, payload, check=True)
            row = {
                "method": cfg["name"],
                "seed": seed,
                "episodes": args.episodes,
                "episode_reward": result["summary"]["episode_reward"],
                "received_energy": result["summary"]["received_energy"],
                "energy_efficiency": result["summary"]["energy_efficiency"],
                "flight_energy": result["summary"]["flight_energy"],
                "trajectory_length": result["summary"]["trajectory_length"],
                "runtime_sec": result["runtime_sec"],
                "trajectory_mode": cfg["trajectory_mode"],
                "observation_mode": cfg["observation_mode"],
                "reward_mode": cfg["reward_mode"],
                "csi_mode": cfg["csi_mode"],
            }
            raw_rows.append(row)

    write_csv(out_dir / "ablation_runs.csv", raw_rows)
    write_json(out_dir / "ablation_runs.json", raw_rows)
    write_csv(out_dir / "ablation_episodes.csv", episodes_rows)
    write_json(out_dir / "ablation_episodes.json", episodes_rows)

    summary_rows: List[Dict[str, Any]] = []
    for cfg in configs:
        rows = [r for r in raw_rows if r["method"] == cfg["name"]]
        summary_rows.append(
            {
                "method": cfg["name"],
                "episode_reward": mean_std_ci([r["episode_reward"] for r in rows]),
                "received_energy": mean_std_ci([r["received_energy"] for r in rows]),
                "energy_efficiency": mean_std_ci([r["energy_efficiency"] for r in rows]),
                "flight_energy": mean_std_ci([r["flight_energy"] for r in rows]),
                "trajectory_length": mean_std_ci([r["trajectory_length"] for r in rows]),
                "runtime_sec": mean_std_ci([r["runtime_sec"] for r in rows]),
            }
        )

    write_json(out_dir / "ablation_summary.json", summary_rows)
    write_csv(
        out_dir / "ablation_summary.csv",
        [
            {
                "method": r["method"],
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

    # Markdown + LaTeX
    md_lines = [
        "| Method | EH | EE | Flight Cost |",
        "|---|---:|---:|---:|",
    ]
    table_rows = []
    for r in summary_rows:
        row = {
            "Method": r["method"],
            "EH": f"{r['episode_reward']['mean']:.4f} ± {r['episode_reward']['std']:.4f}",
            "EE": f"{r['energy_efficiency']['mean']:.4f} ± {r['energy_efficiency']['std']:.4f}",
            "Flight Cost": f"{r['flight_energy']['mean']:.4f} ± {r['flight_energy']['std']:.4f}",
        }
        table_rows.append(row)
        md_lines.append(f"| {row['Method']} | {row['EH']} | {row['EE']} | {row['Flight Cost']} |")
    (out_dir / "ablation_table.md").write_text("\n".join(md_lines) + "\n")
    (out_dir / "ablation_table.tex").write_text(
        rows_to_latex_table(
            table_rows,
            ["Method", "EH", "EE", "Flight Cost"],
            caption="TD3 ablation study.",
            label="tab:ablation",
        )
    )


if __name__ == "__main__":
    main()
