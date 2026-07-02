# Action Space Report

## Summary

The repository uses a normalized action space in `[0, 1]` for all RL scenarios.

The action vector is interpreted as:

1. `tau` — time-splitting / energy-harvesting ratio.
2. `power` — normalized transmit-power control, mapped internally to a physical power range.
3. `theta_R` — RIS phase-control variables, mapped to phases in `[0, 2π]`.
4. Optional learned-trajectory controls — the first three action values are interpreted as UAV motion deltas when `trajectory_mode="learned"`.

## Bounds and semantics

### Fixed-trajectory mode
- Action values remain in `[0, 1]`.
- The environment uses the full action vector for resource allocation.
- The UAV follows the fixed Fermat / KMeans trajectory stored in the dataset.

### Learned-trajectory mode
- The first 3 action values parameterize UAV movement:
  - `dx`, `dy` use `max_step_xy`
  - `dz` uses `max_step_z`
- These values are centered by `(a - 0.5)` so the physical displacement is symmetric around zero.
- The remaining values continue to control `tau`, transmit power, and RIS phases.

## Scenario-specific action dimensions

| Scenario family | Legacy dim | Learned dim |
|---|---:|---:|
| SingleUT-Time | 18 | 21 |
| MultiUT-Time | 20 | 23 |
| SingleUT-Two | 34 | 37 |
| MultiUT-Two | 36 | 39 |

These are the exact shapes used by the environment definitions and are checkpoint-sensitive.

## Compatibility note

Legacy checkpoints remain valid because legacy mode preserves the original action dimensionality exactly.
The learned-trajectory mode is additive and only activates when explicitly requested.
