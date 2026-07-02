# Learnable UAV trajectory mode

The TD3 single-user time-domain environment now supports an optional learned trajectory mode.

Use environment variables before running training scripts:

- `UAV_TRAJECTORY_MODE=learned`
- `UAV_REWARD_MODE=flight_aware` (optional)

Default behavior remains unchanged:
- `UAV_TRAJECTORY_MODE=fixed`
- `UAV_REWARD_MODE=legacy`

The fixed-trajectory baselines still run as before.
