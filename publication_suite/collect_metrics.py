"""Evaluate one model/scenario/configuration and emit JSON metrics."""

from __future__ import annotations

SB3_CUSTOM_OBJECTS = {"lr_schedule": lambda _: 0.0}

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from publication_suite.common import EPS, ensure_dir, repo_root  # noqa: E402


def _reset_env(env, seed: int | None = None):
    if seed is not None and hasattr(env, "seed"):
        try:
            env.seed(seed)
        except Exception:
            pass
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def _step_env(env, action):
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 5:
        next_obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        return next_obs, float(reward), done, info if isinstance(info, dict) else {}
    if isinstance(result, tuple) and len(result) == 4:
        next_obs, reward, done, info = result
        return next_obs, float(reward), bool(done), info if isinstance(info, dict) else {}
    raise RuntimeError(f"Unexpected env.step() return value: {type(result)} / len={len(result) if hasattr(result, '__len__') else 'n/a'}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a single scenario/model configuration.")
    p.add_argument("--scenario-root", required=True, help="Path to one of the algorithm scenario folders.")
    p.add_argument("--algo", required=True, choices=["td3", "ddpg", "sd3"], help="Algorithm name.")
    p.add_argument("--model-path", required=True, help="Path to the saved model (zip or checkpoint prefix).")
    p.add_argument("--trajectory-mode", default="fixed", choices=["fixed", "learned"])
    p.add_argument("--trajectory-source", default="Kmeans", choices=["Kmeans", "Fermat"], help="Fixed-trajectory source for the environment.")
    p.add_argument("--observation-mode", default="legacy", choices=["legacy", "enhanced"])
    p.add_argument("--reward-mode", default="legacy", choices=["legacy", "flight_aware"])
    p.add_argument("--csi-mode", default="perfect", choices=["perfect", "imperfect"])
    p.add_argument("--sigma-csi", type=float, default=0.05)
    p.add_argument("--flight-lambda", type=float, default=1e-3)
    p.add_argument("--num-users", type=int, default=None, help="Optional user count for multi-user envs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def _load_model(algo: str, model_path: str, env, scenario_root: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if algo in {"td3", "ddpg"}:
        try:
            from stable_baselines3 import TD3, DDPG
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("stable_baselines3 is required for TD3/DDPG evaluation.") from exc
        model_cls = TD3 if algo == "td3" else DDPG
        return model_cls.load(model_path, device=device, custom_objects=SB3_CUSTOM_OBJECTS)

    # SD3 custom implementation
    if str(scenario_root) not in sys.path:
        sys.path.insert(0, str(scenario_root))
    import SD3 as sd3_module  # type: ignore

    obs_shape = env.observation_space.shape or (1,)
    action_shape = env.action_space.shape or (1,)
    max_action = float(np.max(env.action_space.high))
    min_action = float(np.min(env.action_space.low))
    model = sd3_module.SD3(
        state_dim=int(obs_shape[0]),
        action_dim=int(action_shape[0]),
        max_action=max_action,
        min_action=min_action,
        device=device,
    )
    model.load(model_path)
    return model


def _predict(model, algo: str, obs, deterministic: bool = True):
    if algo in {"td3", "ddpg"}:
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)
    action = model.select_action(np.asarray(obs, dtype=np.float32))
    return np.asarray(action, dtype=np.float32)


def _trajectory_length(globe_mod) -> float:
    try:
        trajectory = np.asarray(globe_mod.get_value("UAV_Trajectory"), dtype=np.float32)
    except Exception:
        return 0.0
    if trajectory.ndim != 2 or trajectory.shape[0] < 2:
        return 0.0
    diffs = np.diff(trajectory[:, :3], axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def main():
    args = parse_args()
    scenario_root = Path(args.scenario_root).resolve()
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (scenario_root / model_path).resolve()

    if not scenario_root.exists():
        raise FileNotFoundError(f"Scenario root not found: {scenario_root}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    os.chdir(scenario_root)
    if str(scenario_root) not in sys.path:
        sys.path.insert(0, str(scenario_root))

    import gym_foo  # noqa: F401
    import globe  # noqa: F401

    import gymnasium as gym
    env_kwargs: Dict[str, Any] = {
        "Train": False,
        "trajectory_mode": args.trajectory_mode,
        "observation_mode": args.observation_mode,
        "reward_mode": args.reward_mode,
        "csi_mode": args.csi_mode,
        "sigma_csi": args.sigma_csi,
        "flight_lambda": args.flight_lambda,
        "Trajectory_mode": args.trajectory_source,
    }
    if args.num_users is not None and "MultiUT" in str(scenario_root):
        env_kwargs["num_users"] = int(args.num_users)

    env = gym.make("foo-v0", **env_kwargs)
    model = _load_model(args.algo, str(model_path), env, scenario_root)

    episode_rows = []
    start_time = time.perf_counter()
    for ep in range(args.episodes):
        obs = _reset_env(env, seed=args.seed + ep)
        done = False
        total_reward = 0.0
        total_normalized_reward = 0.0
        total_received = 0.0
        total_flight = 0.0
        steps = 0

        while not done:
            action = _predict(model, args.algo, obs, deterministic=True)
            obs, reward, done, info = _step_env(env, action)
            total_normalized_reward += reward
            total_reward += float(info.get("reward", reward))
            total_received += float(info.get("received_energy", 0.0))
            total_flight += float(info.get("flight_energy", 0.0))
            steps += 1

        trajectory_len = _trajectory_length(globe)
        episode_rows.append(
            {
                "episode": ep,
                "episode_reward": total_reward,
                "normalized_episode_reward": total_normalized_reward,
                "received_energy": total_received,
                "energy_efficiency": total_reward / max(total_received, EPS),
                "flight_energy": total_flight,
                "trajectory_length": trajectory_len,
                "steps": steps,
            }
        )

    runtime_sec = time.perf_counter() - start_time
    payload = {
        "algorithm": args.algo,
        "scenario_root": str(scenario_root),
        "model_path": str(model_path),
        "trajectory_mode": args.trajectory_mode,
        "trajectory_source": args.trajectory_source,
        "observation_mode": args.observation_mode,
        "reward_mode": args.reward_mode,
        "csi_mode": args.csi_mode,
        "sigma_csi": args.sigma_csi,
        "flight_lambda": args.flight_lambda,
        "num_users": args.num_users,
        "seed": args.seed,
        "episodes": args.episodes,
        "runtime_sec": runtime_sec,
        "episodes_data": episode_rows,
    }
    summary = {
        "episode_reward": sum(r["episode_reward"] for r in episode_rows) / max(len(episode_rows), 1),
        "received_energy": sum(r["received_energy"] for r in episode_rows) / max(len(episode_rows), 1),
        "energy_efficiency": sum(r["energy_efficiency"] for r in episode_rows) / max(len(episode_rows), 1),
        "flight_energy": sum(r["flight_energy"] for r in episode_rows) / max(len(episode_rows), 1),
        "trajectory_length": sum(r["trajectory_length"] for r in episode_rows) / max(len(episode_rows), 1),
        "normalized_episode_reward": sum(r["normalized_episode_reward"] for r in episode_rows) / max(len(episode_rows), 1),
    }
    payload["summary"] = summary

    ensure_dir(Path(args.output_json).parent)
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
