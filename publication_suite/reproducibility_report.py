from __future__ import annotations


import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import argparse
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from publication_suite.common import ensure_dir, repo_root


def pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return "not_installed"


def main():
    p = argparse.ArgumentParser(description="Generate a reproducibility markdown report.")
    p.add_argument("--output-dir", default="publication_suite_outputs")
    args = p.parse_args()
    root = repo_root()
    out_dir = ensure_dir(root / args.output_dir)

    lines = [
        "# Reproducibility Report",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Python: {sys.version.replace(chr(10), ' ')}",
        f"- Platform: {platform.platform()}",
        "",
        "## Package Versions",
        f"- numpy: {pkg_version('numpy')}",
        f"- matplotlib: {pkg_version('matplotlib')}",
        f"- torch: {pkg_version('torch')}",
        f"- gym: {pkg_version('gym')}",
        f"- gymnasium: {pkg_version('gymnasium')}",
        f"- stable-baselines3: {pkg_version('stable-baselines3')}",
        "",
        "## Example Commands",
        "```bash",
        "python publication_suite/run_matrix.py --episodes 5 --seeds 0 1 2 3 4",
        "python publication_suite/run_ablation.py --episodes 5 --seeds 0 1 2 3 4",
        "python publication_suite/run_robustness.py --algorithm td3 --seeds 0 1 2 3 4 5 6 7 8 9",
        "python publication_suite/run_scalability.py --algorithm td3 --seeds 0 1 2 3 4 --prepare-data",
        "python publication_suite/plot_results.py",
        "python publication_suite/export_tables.py",
        "```",
        "",
        "## Configuration Defaults",
        "- trajectory_mode: fixed or learned",
        "- observation_mode: legacy or enhanced",
        "- reward_mode: legacy or flight_aware",
        "- csi_mode: perfect or imperfect",
    ]
    (out_dir / "REPRODUCIBILITY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
