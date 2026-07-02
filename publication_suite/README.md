# Publication Suite

This folder adds publication-oriented evaluation tooling without changing the training code.

## What it does

- runs the full 16-way experiment matrix
- creates TD3 ablation tables
- performs multi-seed robustness sweeps
- evaluates multi-user scalability
- exports LaTeX tables
- generates publication-style plots
- writes a reproducibility report

## Typical workflow

```bash
python publication_suite/generate_user_trajectories.py --counts 5
python publication_suite/run_matrix.py --episodes 5 --seeds 0 1 2 3 4
python publication_suite/run_ablation.py --episodes 5 --seeds 0 1 2 3 4
python publication_suite/run_robustness.py --algorithm td3 --seeds 0 1 2 3 4 5 6 7 8 9
python publication_suite/run_scalability.py --algorithm td3 --seeds 0 1 2 3 4 --prepare-data
python publication_suite/plot_results.py
python publication_suite/export_tables.py
python publication_suite/reproducibility_report.py
```

## Notes

- The default manifest points to the existing single-user model files in the repository.
- For learned-trajectory experiments, provide the corresponding trained model path through the manifest file.
- The multi-user scalability run supports `N_users = 1, 3, 5` when the matching trajectory CSV files exist.
