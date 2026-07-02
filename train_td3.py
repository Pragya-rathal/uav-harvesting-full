import argparse
import os
import sys
import time

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for gym-only installs
    import gym
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback


def _setup_paths():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(repo_root, "TD3-SingleUT-Time")
    sys.path.insert(0, env_dir)
    os.chdir(env_dir)
    return repo_root


class ProgressPrinter(BaseCallback):
    def __init__(self, check_freq=10000):
        super().__init__()
        self.check_freq = check_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            print(f"[TD3] Steps: {self.num_timesteps} | Elapsed: {elapsed/60:.1f} min")
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train TD3 on the UAV-RIS environment.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="td3_logs")
    parser.add_argument("--model-name", type=str, default="td3_final")

    parser.add_argument(
        "--trajectory-mode",
        type=str,
        default="kmeans",
        choices=("kmeans", "learned"),
        help="Use fixed trajectory baselines or learned trajectory optimization.",
    )
    parser.add_argument(
        "--observation-mode",
        type=str,
        default="legacy",
        choices=("legacy", "enhanced"),
        help="Observation vector variant exposed to the policy.",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="legacy",
        choices=("legacy", "flight_aware"),
        help="Reward shaping mode.",
    )
    parser.add_argument(
        "--csi-mode",
        type=str,
        default="perfect",
        choices=("perfect", "imperfect"),
        help="Channel state information mode.",
    )
    parser.add_argument(
        "--sigma-csi",
        type=float,
        default=0.05,
        help="Standard deviation of CSI perturbation noise in imperfect mode.",
    )
    parser.add_argument(
        "--flight-lambda",
        type=float,
        default=1e-3,
        help="Flight-energy penalty weight used in flight-aware reward mode.",
    )
    parser.add_argument(
        "--max-step-xy",
        type=float,
        default=10.0,
        help="Maximum learned trajectory step in x/y.",
    )
    parser.add_argument(
        "--max-step-z",
        type=float,
        default=4.0,
        help="Maximum learned trajectory step in z.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = _setup_paths()
    import gym_foo  # noqa: F401

    log_dir = os.path.join(repo_root, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    env = gym.make(
        "foo-v0",
        Train=True,
        trajectory_mode=args.trajectory_mode,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        csi_mode=args.csi_mode,
        sigma_csi=args.sigma_csi,
        flight_lambda=args.flight_lambda,
        max_step_xy=args.max_step_xy,
        max_step_z=args.max_step_z,
    )
    env.reset(seed=args.seed)

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=256,
        verbose=1,
        device=device,
        seed=args.seed,
    )

    print("Starting TD3 training...")
    callback = ProgressPrinter(check_freq=10000)
    model.learn(total_timesteps=args.timesteps, log_interval=10, callback=callback)

    save_path = os.path.join(log_dir, args.model_name)
    model.save(save_path)
    print(f"Saved TD3 model to {save_path}.zip")


if __name__ == "__main__":
    main()
