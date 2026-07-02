# Research Upgrade Notes

This repository now supports the following optional environment modes while preserving the default baseline behavior:

- `UAV_TRAJECTORY_MODE`: `fixed` (default) or `learned`
- `UAV_OBSERVATION_MODE`: `legacy` (default) or `enhanced`
- `UAV_REWARD_MODE`: `legacy` (default) or `flight_aware`
- `UAV_CSI_MODE`: `perfect` (default) or `imperfect`
- `UAV_CSI_SIGMA`: standard deviation for CSI perturbation (default `0.05`)
- `UAV_FLIGHT_LAMBDA`: flight-energy penalty weight (default `1e-3`)

Default mode reproduces the original behavior.
