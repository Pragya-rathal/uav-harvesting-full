# State Representation Report

## Legacy observation mode

Legacy mode returns the original 2-D observation used by the repository:

- AP-to-UAV distance
- RIS-to-user distance

This is the compatibility-preserving mode for old checkpoints and baselines.

## Enhanced observation mode

Enhanced mode adds only simulator-available variables, without inventing new physics.

The enhanced state includes:

- normalized step index
- UAV position `(x, y, z)`
- radio-state distances
- current reward
- received energy
- learned-trajectory flag
- imperfect-CSI flag
- per-user location and UAV-user distance for each active user

## Dimension rule

The environment pads or truncates the enhanced vector to a fixed size:

`enhanced_obs_dim = 10 + 4 * len(active_users)`

Examples:

- 1 user → 14 dims
- 3 users → 22 dims
- 5 users → 30 dims

## Checkpoint compatibility

- Legacy mode preserves the exact original observation shape.
- Enhanced mode is opt-in and additive.
- No legacy checkpoint is forced to consume the enhanced state.
