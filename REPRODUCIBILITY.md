# Reproducibility

This repository now includes a reproducibility generator in `publication_suite/reproducibility_report.py`.

## Recommended environment

- Python 3.11 or 3.13 with working CUDA if GPU training is desired
- `torch`
- `stable-baselines3`
- `gymnasium==0.29.1`
- `gym==0.26.2`
- `shimmy`
- `numpy<2.0`
- `matplotlib`

## Core commands

```bash
python train_sac.py --timesteps 200000 --model-name sac_final --log-dir sac_logs
python train_td3.py --timesteps 200000 --model-name td3_final --log-dir td3_logs
python train_ddpg.py --timesteps 200000 --model-name ddpg_final --log-dir ddpg_logs
python eval_sweep.py --algo all --k-values 0.5 5.0 --episodes-per-pt 5
python publication_suite/run_matrix.py --episodes 5 --seeds 0 1 2 3 4
python publication_suite/run_ablation.py --episodes 5 --seeds 0 1 2 3 4
python publication_suite/run_robustness.py --algorithm td3 --seeds 0 1 2 3 4 5 6 7 8 9
python publication_suite/run_scalability.py --algorithm td3 --seeds 0 1 2 3 4 --prepare-data
python publication_suite/plot_results.py
python publication_suite/export_tables.py
```

## Validation status

The repository is now syntactically valid and the publication-suite scripts can be executed directly as files.
