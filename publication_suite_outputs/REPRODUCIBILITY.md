# Reproducibility Report

- Generated: 2026-06-21T14:42:46.142836+00:00
- Python: 3.13.5 (main, May  5 2026, 21:05:52) [GCC 14.2.0]
- Platform: Linux-4.4.0-x86_64-with-glibc2.41

## Package Versions
- numpy: 2.3.5
- matplotlib: 3.10.8
- torch: 2.10.0+cpu
- gym: not_installed
- gymnasium: not_installed
- stable-baselines3: not_installed

## Example Commands
```bash
python publication_suite/run_matrix.py --episodes 5 --seeds 0 1 2 3 4
python publication_suite/run_ablation.py --episodes 5 --seeds 0 1 2 3 4
python publication_suite/run_robustness.py --algorithm td3 --seeds 0 1 2 3 4 5 6 7 8 9
python publication_suite/run_scalability.py --algorithm td3 --seeds 0 1 2 3 4 --prepare-data
python publication_suite/plot_results.py
python publication_suite/export_tables.py
```

## Configuration Defaults
- trajectory_mode: fixed or learned
- observation_mode: legacy or enhanced
- reward_mode: legacy or flight_aware
- csi_mode: perfect or imperfect
