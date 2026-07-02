# Scalability Report

## User-count sweep supported by the publication suite

The benchmark supports evaluation at:

- `N_users = 1`
- `N_users = 3`
- `N_users = 5`

## How scaling is implemented

- The publication suite passes `num_users` into the environment when multi-user mode is active.
- `generate_user_trajectories.py` can prepare supporting user-trajectories for the user-count sweep.
- Results are aggregated into CSV / JSON / LaTeX-friendly outputs.

## Metrics to report

- harvested energy
- received energy
- energy efficiency
- runtime
- flight energy

## Validation placeholder

Insert the final tables and plots after the runs complete.
