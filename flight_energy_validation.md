# Flight-Energy Validation Report

## Supported reward modes

- `reward_mode="legacy"`
- `reward_mode="flight_aware"`

## Flight-energy model currently implemented

The environment subtracts a flight-energy penalty when `reward_mode="flight_aware"`.

Current implementation:

`flight_energy = alpha_xy * (dx² + dy²) + alpha_z * (dz²)`

with:
- `alpha_xy = 1.0`
- `alpha_z = 1.5`

The reward becomes:

`reward_flight_aware = original_reward - lambda_flight * flight_energy`

## Behavior by trajectory mode

### Learned trajectory
The action controls UAV displacement, and the penalty is applied directly to the action-induced motion delta.

### Fixed trajectory
The penalty is computed from consecutive stored trajectory points.

## Compatibility guarantee

- `reward_mode="legacy"` leaves the original reward path intact.
- Existing checkpoints continue to load.
- Baselines remain reproducible.

## Validation placeholder

After the long training runs finish, insert:
- reward curves
- flight-energy curves
- final reward deltas
- efficiency trade-off plots
