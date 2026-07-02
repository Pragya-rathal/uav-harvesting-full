# Novelty Validation Smoke Report

## Environment
- Repo: UAV-RIS_EnergyHarvesting-main
- Environment audited: `TD3-SingleUT-Time/gym_foo/envs/foo_env.py`
- Evaluation method: short deterministic rollouts with fixed action sequences and multiple seeds

## Key results

### Observation mode
- Legacy observation dimension: `2`
- Enhanced observation dimension: `22`
- Mean L2 difference between legacy and enhanced observations over the first 20 timesteps: `73.9869`

### Flight-aware reward
Using the same actions and seed:
- Legacy mean reward: `0.0`
- Flight-aware mean reward: `-0.0018936242`
- Mean flight-energy penalty term: `1.8936242`

### Imperfect CSI
Across 10 seeds with the same action sequence:
- Mean absolute difference in received energy between perfect and imperfect CSI: `0.1070 ± 0.0082` (std across seeds)
- Perfect CSI mean received energy: `4.2477`
- Imperfect CSI mean received energy: `4.2487`

Interpretation:
- The perturbation is active.
- The mean received energy changes only slightly because the perturbation is approximately zero-mean.
- Variance increases slightly under imperfect CSI.

### Learned trajectory
- Fixed trajectory path length in the smoke test: `0.0`
- Learned trajectory path length in the smoke test: `53.98`
- Fixed start position: `[0.0, 0.0, 20.0]`
- Learned start position: `[6.0, 13.0, 20.0]`

## Root cause of identical-looking plots
The root training/evaluation scripts originally did not expose the novelty flags, so experiments defaulted to legacy behavior.
That exposure is now added in the patched scripts.

A second reason is that the Pt sweep plots can still look similar across algorithms because the evaluation loop externally sets transmit power and the environment is strongly channel-dominated.
