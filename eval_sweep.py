import argparse
import os
import sys

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


def _setup_paths():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(repo_root, "TD3-SingleUT-Time")
    sys.path.insert(0, env_dir)
    os.chdir(env_dir)
    return repo_root


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SAC/TD3/DDPG over Pt and compare multiple Rician K factors in one run."
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "td3", "ddpg", "all"],
        help="RL algorithm used for loading the trained policy. Use 'all' to evaluate SAC, TD3, and DDPG in one run.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model zip. If omitted, defaults to <algo>_logs/<algo>_final.zip.",
    )
    parser.add_argument(
        "--k-values",
        type=float,
        nargs="+",
        default=[0.5, 5.0],
        help="Rician K values to evaluate in a single run.",
    )
    parser.add_argument(
        "--episodes-per-pt",
        type=int,
        default=5,
        help="Evaluation episodes for each Pt value.",
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

    default_model_paths = {
        "sac": "sac_logs/sac_final.zip",
        "td3": "td3_logs/td3_final.zip",
        "ddpg": "ddpg_logs/ddpg_final.zip",
    }
    model_loaders = {"sac": SAC, "td3": TD3, "ddpg": DDPG}
    algo_list = ["sac", "td3", "ddpg"] if args.algo == "all" else [args.algo]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pt_values = [10, 15, 20, 25, 30]

    for algo_name in algo_list:
        model_path = args.model_path or default_model_paths[algo_name]
        if not os.path.isabs(model_path):
            model_path = os.path.join(repo_root, model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Train the selected algorithm first."
            )

        model = model_loaders[algo_name].load(model_path, device=device, custom_objects=SB3_CUSTOM_OBJECTS)
        all_results = {}

        for k_value in args.k_values:
            env = gym.make(
                "foo-v0",
                Train=False,
                RicianK=float(k_value),
                trajectory_mode=args.trajectory_mode,
                observation_mode=args.observation_mode,
                reward_mode=args.reward_mode,
                csi_mode=args.csi_mode,
                sigma_csi=args.sigma_csi,
                flight_lambda=args.flight_lambda,
                max_step_xy=args.max_step_xy,
                max_step_z=args.max_step_z,
            )
            env.reset(seed=0)

            avg_sinr = []
            avg_sumrate = []

            for pt_dbm in pt_values:
                sinr_runs = []
                sumrate_runs = []
                for _ in range(args.episodes_per_pt):
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

                        obs, reward, terminated, truncated, info = env.step(action)

                    sinr_runs.append(float(np.mean(episode_sinr)))
                    sumrate_runs.append(float(np.mean(episode_sumrate)))

                avg_sinr.append(float(np.mean(sinr_runs)))
                avg_sumrate.append(float(np.mean(sumrate_runs)))
                print(
                    f"Algo={algo_name.upper()}, K={k_value:g}, Pt={pt_dbm} dBm | "
                    f"SINR={avg_sinr[-1]:.3f} | SumRate={avg_sumrate[-1]:.3f}"
                )

            all_results[float(k_value)] = {
                "sinr": avg_sinr,
                "sumrate": avg_sumrate,
            }

        plt.figure()
        for k_value, metrics in all_results.items():
            plt.plot(pt_values, metrics["sinr"], marker="o", label=f"K={k_value:g}")
        plt.xlabel("Transmit Power Pt (dBm)")
        plt.ylabel("Average SINR (dB)")
        plt.title("SINR vs Transmit Power")
        plt.grid(True)
        plt.legend()
        prefix = algo_name.upper()
        sinr_path = os.path.join(repo_root, f"SINR_vs_Pt_{prefix}_Kcompare.png")
        plt.savefig(sinr_path, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        for k_value, metrics in all_results.items():
            plt.plot(pt_values, metrics["sumrate"], marker="o", label=f"K={k_value:g}")
        plt.xlabel("Transmit Power Pt (dBm)")
        plt.ylabel("Average Sum-Rate")
        plt.title("Sum-Rate vs Transmit Power")
        plt.grid(True)
        plt.legend()
        sumrate_path = os.path.join(repo_root, f"SumRate_vs_Pt_{prefix}_Kcompare.png")
        plt.savefig(sumrate_path, dpi=300, bbox_inches="tight")
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for k_value, metrics in all_results.items():
            axes[0].plot(pt_values, metrics["sinr"], marker="o", label=f"K={k_value:g}")
            axes[1].plot(pt_values, metrics["sumrate"], marker="o", label=f"K={k_value:g}")
        axes[0].set_xlabel("Transmit Power Pt (dBm)")
        axes[0].set_ylabel("Average SINR (dB)")
        axes[0].set_title("SINR vs Pt")
        axes[0].grid(True)
        axes[0].legend()

        axes[1].set_xlabel("Transmit Power Pt (dBm)")
        axes[1].set_ylabel("Average Sum-Rate")
        axes[1].set_title("Sum-Rate vs Pt")
        axes[1].grid(True)
        axes[1].legend()

        fig.tight_layout()
        combined_path = os.path.join(repo_root, f"SINR_SumRate_vs_Pt_{prefix}_Kcompare.png")
        fig.savefig(combined_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved plots to {sinr_path}, {sumrate_path}, and {combined_path}")



if __name__ == "__main__":
    main()
