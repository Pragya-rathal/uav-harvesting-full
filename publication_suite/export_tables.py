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

from publication_suite.common import ensure_dir, format_pm, repo_root


def read_csv_rows(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def latex_table(rows, columns, caption, label):
    head = " & ".join(columns) + r" \\"
    body = "\n".join(" & ".join(str(r[c]) for c in columns) + r" \\" for r in rows)
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\begin{tabular}{" + "l" * len(columns) + r"}",
            r"\hline",
            head,
            r"\hline",
            body,
            r"\hline",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{table}",
        ]
    )


def export_summary(csv_path: Path, out_path: Path, key_col: str, value_cols):
    if not csv_path.exists():
        return
    rows = read_csv_rows(csv_path)
    if not rows:
        return
    out_rows = []
    for r in rows:
        row = {key_col.title().replace("_", " "): r[key_col]}
        for base_col in value_cols:
            mean = float(r[base_col])
            std = float(r[base_col + "_std"])
            row[base_col.replace("_", " ").title()] = format_pm(mean, std, precision=4)
        out_rows.append(row)
    out_path.write_text(latex_table(out_rows, list(out_rows[0].keys()), out_path.stem.replace("_", " ").title(), f"tab:{out_path.stem}"))


def main():
    p = argparse.ArgumentParser(description="Export IEEE-ready LaTeX tables.")
    p.add_argument("--input-dir", default="publication_suite_outputs")
    args = p.parse_args()
    root = repo_root()
    in_dir = root / args.input_dir

    ensure_dir(in_dir)

    export_summary(
        in_dir / "matrix" / "matrix_summary.csv",
        in_dir / "results_table.tex",
        "algorithm",
        ["episode_reward", "received_energy", "energy_efficiency", "flight_energy"],
    )
    export_summary(
        in_dir / "ablation" / "ablation_summary.csv",
        in_dir / "ablation_table.tex",
        "method",
        ["episode_reward", "received_energy", "energy_efficiency", "flight_energy"],
    )
    export_summary(
        in_dir / "scalability" / "scalability_summary.csv",
        in_dir / "scalability_table.tex",
        "user_count",
        ["episode_reward", "received_energy", "energy_efficiency", "flight_energy"],
    )


if __name__ == "__main__":
    main()
