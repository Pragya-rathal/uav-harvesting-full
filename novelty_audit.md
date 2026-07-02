# Novelty Audit and Validation Report

Repository: UAV-RIS_EnergyHarvesting-main

## Validation scope

I audited the code paths that implement:
- observation mode
- reward mode / flight-energy penalty
- CSI mode
- learned trajectory mode
- checkpoint loading and model distinctness

I also ran short deterministic smoke-rollouts against the `TD3-SingleUT-Time` environment with a fixed action sequence to verify that the new modes change behavior.

## Code locations

### Observation mode
- `TD3-SingleUT-Time/gym_foo/envs/foo_env.py`
- `TD3-MultiUT-Time/gym_foo/envs/foo_env.py`
- `DDPG-*`, `SD3-*`, `Exhaustive-*` environment copies

Relevant logic:
- `self.observation_mode`
- `self._build_observation(...)`
- legacy observation: 2 values
- enhanced observation: step index, UAV position, radio state, reward, received energy, trajectory/csi flags, per-user coordinates/distances

### Reward mode / flight energy
- same env files as above

Relevant logic:
- `self.reward_mode`
- `self.flight_lambda`
- `self._flight_energy(delta)`
- `reward -= self.flight_lambda * self._flight_energy(...)`

### CSI robustness
- same env files as above

Relevant logic:
- `self.csi_mode`
- `self.sigma_csi`
- `self._csi_perturb(link)`

Decision path:
- `EH(...)`
- `capacity(...)`
- `_csi_perturb(...)` is applied inside the channel / throughput computation path

### Learned trajectory mode
- same env files as above

Relevant logic:
- `self.trajectory_mode`
- `learned = self.trajectory_mode == "learned"`
- learned mode uses action[0:3] for UAV motion
- fixed mode uses precomputed UAV trajectories from the dataset

### Training / evaluation exposure
Before patching, the root-level training/evaluation scripts did not expose the novelty flags. I patched:
- `train_sac.py`
- `train_td3.py`
- `train_ddpg.py`
- `train_ppo.py`
- `eval_sweep.py`
- `eval_compare.py`
- `eval_plots.py`
- `rician_k_sweep.py`

These now accept and forward:
- `--trajectory-mode`
- `--observation-mode`
- `--reward-mode`
- `--csi-mode`
- `--sigma-csi`
- `--flight-lambda`
- `--max-step-xy`
- `--max-step-z`

## Smoke validation results

I ran short deterministic rollouts with the extracted environment code.

### 1) Observation mode changes the state vector
- legacy observation dimension: `2`
- enhanced observation dimension: `22`

Mean L2 difference between legacy and enhanced observations over the first 20 timesteps:
- `73.9869`

This confirms the policy input changes materially when enhanced observations are enabled.

### 2) Flight-aware reward changes the reward signal
Using the same action sequence and seed:
- legacy mean reward: `0.0`
- flight-aware mean reward: `-0.0018936242`

Mean flight-energy term:
- `1.8936242`

This shows the flight-energy penalty is active and changes the return. The percentage change is not meaningful when the legacy reward is exactly zero.

### 3) Imperfect CSI changes the channel-derived outputs
Using the same seed and action sequence:
- mean absolute difference in received energy (perfect vs imperfect CSI): `0.3067`

This confirms `_csi_perturb(...)` is affecting the channel/throughput path.

### 4) Learned trajectory changes mobility behavior
With the same seed:
- fixed trajectory start position: `[0.0, 0.0, 20.0]`
- learned trajectory start position: `[6.0, 13.0, 20.0]`

Over the short rollout:
- fixed path length: `0.0`
- learned path length: `53.98`

This confirms the learned-trajectory pathway is active and changes UAV motion.

## Plot-divergence root cause found

The strongest code-level reason the curves can look overly similar is that the root training/evaluation scripts originally did not expose the novelty flags, so experiments defaulted to legacy settings.

That is now fixed in the patched scripts listed above.

A second reason is that the evaluation scripts sweep transmit power externally, so some plots are naturally dominated by the Pt sweep and may look similar across algorithms unless the trained policies are sufficiently different.

## Checkpoint sanity

Model artifacts present in the repository are distinct by file hash:
- TD3 checkpoints are separate from DDPG checkpoints
- SD3 checkpoints exist in the SD3 subdirectories as their own actor/critic files

This means the repository does not appear to reuse one checkpoint file for all algorithms.

## Bottom line

The novelty mechanisms are present in the environment code and the root scripts now expose them.
The short smoke tests confirm that:
- enhanced observations change the state vector,
- flight-aware reward changes returns,
- imperfect CSI changes channel outputs,
- learned trajectory changes UAV motion.

The remaining step for the paper is to run the full training / ablation / seed sweep experiments and export the final plots and tables from those runs.
