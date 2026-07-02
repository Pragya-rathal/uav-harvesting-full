import argparse
import os
import shutil
import sys
import time

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback for gym-only installs
    import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
EPS = 1e-8
SB3_CUSTOM_OBJECTS = {"lr_schedule": lambda _: 0.0}

from stable_baselines3 import DDPG, SAC, TD3
EPS = 1e-8

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
EPS = 1e-8

from stable_baselines3.common.monitor import Monitor


def _setup_paths():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(repo_root, "TD3-SingleUT-Time")
    sys.path.insert(0, env_dir)
    os.chdir(env_dir)
    return repo_root


class ProgressPrinter(BaseCallback):
    def __init__(self, algo_name, check_freq=10000):
        super().__init__()
        self.algo_name = algo_name
        self.check_freq = check_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = time.time() - self.start_time
            print(
                f"[{self.algo_name}] Steps: {self.num_timesteps} | Elapsed: {elapsed/60:.1f} min"
            )
        return True


class EpisodeRewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True


def _quantize_theta(action_slice, levels=8):
    phase_levels = np.linspace(0, 2 * np.pi, levels, endpoint=False)
    phase_indices = np.minimum((action_slice * levels).astype(int), levels - 1)
    return phase_levels[phase_indices]


def compute_sinr_sumrate(env, tau, power_1, theta_r, l_u, l_ap, ut_0):
    awgn = globe.get_value("AWGN")
    bw = globe.get_value("BW")
    bs_z = globe.get_value("BS_Z")
    ris_l = globe.get_value("RIS_L")

    g_br = env.pl_BR(l_u, l_ap)
    small_fading_g = env.SmallFading_G(bs_z, ris_l)
    g = np.ones((bs_z, ris_l)) * g_br * small_fading_g
    coefficients = np.diag(np.exp(1j * theta_r))

    h_ru = env.Channel_RU(l_u, ut_0, bs_z, ris_l)
    ut_link = np.linalg.multi_dot([g, coefficients, h_ru])
    signal_ut = np.sum(np.abs(ut_link * np.conjugate(ut_link))) * power_1

    sinr_linear = signal_ut / max(float(awgn), EPS)
    sinr_db = 10 * np.log10(max(float(sinr_linear), EPS))
    if sinr_linear > 0:
        sum_rate = bw * np.log2(1 + max(float(sinr_linear), EPS)) * (1 - tau)
    else:
        sum_rate = 0.0

    return sinr_db, sum_rate


def evaluate_model(model, env, pt_values, episodes_per_pt):
    avg_sinr = []
    avg_sumrate = []
    for pt_dbm in pt_values:
        sinr_runs = []
        sumrate_runs = []
        for _ in range(episodes_per_pt):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            episode_sinr = []
            episode_sumrate = []
            while not (terminated or truncated):
                action, _states = model.predict(obs, deterministic=True)
                action = np.array(action, dtype=np.float32)
                action[1] = pt_dbm / 30.0

                step = globe.get_value("step")
                t = globe.get_value("t")
                if step < t - 1:
                    l_u = globe.get_value("UAV_Trajectory")[step + 1]
                    ut_0 = globe.get_value("UT_0")[step + 1]
                else:
                    l_u = globe.get_value("UAV_Trajectory")[step]
                    ut_0 = globe.get_value("UT_0")[step]
                l_ap = globe.get_value("L_AP")

                tau = action[0]
                power_1 = 10 ** (((action[1] - 1) * 30 / 10) + 3)
                theta_r = _quantize_theta(action[2:])

                sinr_db, sum_rate = compute_sinr_sumrate(
                    env, tau, power_1, theta_r, l_u, l_ap, ut_0
                )
                episode_sinr.append(sinr_db)
                episode_sumrate.append(sum_rate)

                obs, _reward, terminated, truncated, _info = env.step(action)

            sinr_runs.append(float(np.mean(episode_sinr)))
            sumrate_runs.append(float(np.mean(episode_sumrate)))

        avg_sinr.append(float(np.mean(sinr_runs)))
        avg_sumrate.append(float(np.mean(sumrate_runs)))

    return avg_sinr, avg_sumrate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep Rician K and compare TD3/DDPG/SAC under fixed settings."
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument(
        "--chunk-timesteps",
        type=int,
        default=None,
        help="Number of timesteps to train per chunk (defaults to --timesteps).",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Path to a model checkpoint to resume from.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-per-pt", type=int, default=5)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--algo",
        type=str,
        default="all",
        choices=("all", "td3", "ddpg", "sac"),
        help="Train a single algorithm or all (default: all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Force training device (auto uses CUDA if available).",
    )

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
    global globe
    import globe

    chunk_timesteps = args.chunk_timesteps or args.timesteps
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Weak (0.5) and medium (1) Rician K; baseline (5) excluded. All other parameters fixed for fairness.
    K_values = [0.5, 1]
    pt_values = [10, 15, 20, 25, 30]
    algo_specs = {
        "TD3": TD3,
        "DDPG": DDPG,
        "SAC": SAC,
    }
    if args.algo != "all":
        algo_key = args.algo.upper()
        algo_specs = {algo_key: algo_specs[algo_key]}

    for k_value in K_values:
        k_dir = os.path.join(repo_root, args.results_dir, f"K_{k_value}")
        os.makedirs(k_dir, exist_ok=True)

        sinr_series = {}
        sumrate_series = {}
        reward_series = {}

        for algo_name, algo_cls in algo_specs.items():
            algo_dir = os.path.join(k_dir, algo_name)
            os.makedirs(algo_dir, exist_ok=True)

            env = gym.make(
        "foo-v0",
        Train=True,
        RicianK=k_value,
        trajectory_mode=args.trajectory_mode,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        csi_mode=args.csi_mode,
        sigma_csi=args.sigma_csi,
        flight_lambda=args.flight_lambda,
        max_step_xy=args.max_step_xy,
        max_step_z=args.max_step_z,
    )
            env = Monitor(env)
            env.reset(seed=args.seed)

            resume_path = None
            if args.resume_path:
                resume_path = args.resume_path.format(
                    algo=algo_name.lower(), k=k_value
                )
                if not os.path.exists(resume_path):
                    raise FileNotFoundError(
                        f"Resume checkpoint not found: {resume_path}"
                    )

            if resume_path:
                print(f"Resuming from checkpoint: {resume_path}")
                model = algo_cls.load(resume_path, env=env, device=device, custom_objects=SB3_CUSTOM_OBJECTS)
            else:
                print("Starting new training from scratch...")
                model = algo_cls(
                    "MlpPolicy",
                    env,
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    batch_size=256,
                    verbose=1,
                    tensorboard_log=os.path.join(algo_dir, "tensorboard"),
                    device=device,
                    seed=args.seed,
                )

            print(f"Starting {algo_name} training with K={k_value}...")
            reward_logger = EpisodeRewardLogger()
            callback = CallbackList(
                [ProgressPrinter(algo_name, check_freq=10000), reward_logger]
            )
            model.learn(
                total_timesteps=chunk_timesteps,
                log_interval=10,
                callback=callback,
                reset_num_timesteps=False,
            )

            checkpoint_dir = os.path.join(repo_root, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            current_total_steps = model.num_timesteps
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{algo_name.lower()}_k{k_value}_step{current_total_steps}.zip",
            )
            latest_path = os.path.join(
                checkpoint_dir, f"{algo_name.lower()}_k{k_value}_latest.zip"
            )
            model.save(checkpoint_path)
            model.save(latest_path)
            print(f"Chunk completed. Saved to: {checkpoint_path}")

            save_path = os.path.join(algo_dir, f"{algo_name.lower()}_final")
            model.save(save_path)
            reward_series[algo_name] = reward_logger.episode_rewards

            eval_env = gym.make(
        "foo-v0",
        Train=False,
        RicianK=k_value,
        trajectory_mode=args.trajectory_mode,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        csi_mode=args.csi_mode,
        sigma_csi=args.sigma_csi,
        flight_lambda=args.flight_lambda,
        max_step_xy=args.max_step_xy,
        max_step_z=args.max_step_z,
    )
            eval_env.reset(seed=args.seed)
            avg_sinr, avg_sumrate = evaluate_model(
                model, eval_env, pt_values, args.episodes_per_pt
            )
            sinr_series[algo_name] = avg_sinr
            sumrate_series[algo_name] = avg_sumrate
            eval_env.close()
            env.close()

        plt.figure()
        for algo_name, values in sinr_series.items():
            plt.plot(pt_values, values, marker="o", label=algo_name)
        plt.xlabel("Transmit Power Pt (dBm)")
        plt.ylabel("Average SINR (dB)")
        plt.title(f"SINR vs Transmit Power (K={k_value})")
        plt.grid(True)
        plt.legend()
        sinr_path = os.path.join(k_dir, "sinr_vs_pt.png")
        plt.savefig(sinr_path, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        for algo_name, values in sumrate_series.items():
            plt.plot(pt_values, values, marker="o", label=algo_name)
        plt.xlabel("Transmit Power Pt (dBm)")
        plt.ylabel("Average Sum-Rate")
        plt.title(f"Sum-Rate vs Transmit Power (K={k_value})")
        plt.grid(True)
        plt.legend()
        sumrate_path = os.path.join(k_dir, "sumrate_vs_pt.png")
        plt.savefig(sumrate_path, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        for algo_name, rewards in reward_series.items():
            episodes = np.arange(1, len(rewards) + 1)
            plt.plot(episodes, rewards, label=algo_name)
        plt.xlabel("Episode")
        plt.ylabel("Training Reward")
        plt.title(f"Training Reward vs Episodes (K={k_value})")
        plt.grid(True)
        plt.legend()
        reward_path = os.path.join(k_dir, "training_reward.png")
        plt.savefig(reward_path, dpi=300, bbox_inches="tight")
        plt.close()

        for algo_name in algo_specs.keys():
            algo_dir = os.path.join(k_dir, algo_name)
            shutil.copyfile(sinr_path, os.path.join(algo_dir, "sinr_vs_pt.png"))
            shutil.copyfile(sumrate_path, os.path.join(algo_dir, "sumrate_vs_pt.png"))
            shutil.copyfile(reward_path, os.path.join(algo_dir, "training_reward.png"))

        print(f"Saved plots for K={k_value} under {k_dir}.")


if __name__ == "__main__":
    main()
