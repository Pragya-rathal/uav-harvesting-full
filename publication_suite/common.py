"""Shared helpers for publication-ready evaluation scripts."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


EPS = 1e-8


@dataclass(frozen=True)
class RunConfig:
    algorithm: str
    scenario_root: str
    model_path: str
    trajectory_mode: str = "fixed"
    observation_mode: str = "legacy"
    reward_mode: str = "legacy"
    csi_mode: str = "perfect"
    sigma_csi: float = 0.05
    flight_lambda: float = 1e-3
    num_users: int | None = None
    seed: int = 0
    episodes: int = 5
    deterministic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: os.PathLike[str] | str, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_csv(path: os.PathLike[str] | str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def read_csv(path: os.PathLike[str] | str) -> List[Dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return list(csv.DictReader(f))


def mean_std_ci(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "ci95": 0.0, "n": 0.0}
    n = len(vals)
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "min": min(vals),
        "max": max(vals),
        "ci95": ci95,
        "n": float(n),
    }


def format_pm(mean: float, std: float, precision: int = 4) -> str:
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def build_default_manifest() -> Dict[str, Dict[str, Any]]:
    """Default scenario/model paths used if the user does not provide a manifest."""
    root = repo_root()
    return {
        "td3": {
            "scenario_root": str(root / "TD3-SingleUT-Time"),
            "model_path": str(root / "TD3-SingleUT-Time" / "td3_SingleUT_Time.zip"),
            "kind": "sb3",
        },
        "ddpg": {
            "scenario_root": str(root / "DDPG-SingleUT-Time"),
            "model_path": str(root / "DDPG-SingleUT-Time" / "ddpg_SingleUT_Time.zip"),
            "kind": "sb3",
        },
        "sd3": {
            "scenario_root": str(root / "SD3-SingleUT-Time"),
            "model_path": str(root / "SD3-SingleUT-Time" / "checkpoints" / "models" / "model"),
            "kind": "sd3",
        },
    }


def build_matrix_configs() -> List[Dict[str, str]]:
    configs = []
    for trajectory_mode in ("fixed", "learned"):
        for observation_mode in ("legacy", "enhanced"):
            for reward_mode in ("legacy", "flight_aware"):
                for csi_mode in ("perfect", "imperfect"):
                    configs.append(
                        {
                            "trajectory_mode": trajectory_mode,
                            "observation_mode": observation_mode,
                            "reward_mode": reward_mode,
                            "csi_mode": csi_mode,
                        }
                    )
    return configs


def build_td3_ablation_configs() -> List[Dict[str, str]]:
    return [
        {
            "name": "A0_fixed_baseline",
            "trajectory_mode": "fixed",
            "observation_mode": "legacy",
            "reward_mode": "legacy",
            "csi_mode": "perfect",
        },
        {
            "name": "A1_learned_trajectory",
            "trajectory_mode": "learned",
            "observation_mode": "legacy",
            "reward_mode": "legacy",
            "csi_mode": "perfect",
        },
        {
            "name": "A2_learned_plus_enhanced_state",
            "trajectory_mode": "learned",
            "observation_mode": "enhanced",
            "reward_mode": "legacy",
            "csi_mode": "perfect",
        },
        {
            "name": "A3_learned_plus_state_plus_flight",
            "trajectory_mode": "learned",
            "observation_mode": "enhanced",
            "reward_mode": "flight_aware",
            "csi_mode": "perfect",
        },
        {
            "name": "A4_full_model",
            "trajectory_mode": "learned",
            "observation_mode": "enhanced",
            "reward_mode": "flight_aware",
            "csi_mode": "imperfect",
        },
    ]


def build_scalability_user_counts() -> List[int]:
    return [1, 3, 5]


def flatten_episode_metrics(config: Mapping[str, Any], episode_metrics: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ep in episode_metrics:
        row = dict(config)
        row.update(ep)
        rows.append(row)
    return rows


def run_worker_subprocess(
    worker_script: os.PathLike[str] | str,
    payload: Mapping[str, Any],
    python_exe: str | None = None,
    check: bool = True,
) -> Dict[str, Any]:
    """Run a worker script in a subprocess and return the parsed JSON payload."""
    import subprocess
    import sys
    import tempfile

    worker_script = str(worker_script)
    python_exe = python_exe or sys.executable
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        out_json = tmp.name
    cmd = [
        python_exe,
        worker_script,
        "--output-json",
        out_json,
    ]
    for k, v in payload.items():
        arg = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(arg)
        elif v is None:
            continue
        elif isinstance(v, (list, tuple)):
            for item in v:
                cmd.extend([arg, str(item)])
        else:
            cmd.extend([arg, str(v)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if check:
            raise RuntimeError(
                f"Worker failed with exit code {proc.returncode}.\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        return {
            "status": "failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    payload_out = json.loads(Path(out_json).read_text())
    try:
        Path(out_json).unlink(missing_ok=True)
    except Exception:
        pass
    return payload_out


def rows_to_latex_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], caption: str = "", label: str = "") -> str:
    header = " & ".join(columns) + r" \\"
    body = "\n".join(" & ".join(str(row.get(col, "")) for col in columns) + r" \\" for row in rows)
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{" + "l" * len(columns) + r"}",
        r"\hline",
        header,
        r"\hline",
        body,
        r"\hline",
        r"\end{tabular}",
        f"\\caption{{{caption}}}" if caption else r"\caption{}",
        f"\\label{{{label}}}" if label else r"\label{}",
        r"\end{table}",
    ])
