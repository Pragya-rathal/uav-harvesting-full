# SAC Paper Notes (Current Repo Snapshot)

This note summarizes the SAC-related environment/model equations, script workflows, plotting pipeline, and key repo-level SAC changes visible on the current branch.

## 1) Environment and system equations used by SAC

Environment used by the modern SAC script is `TD3-SingleUT-Time/gym_foo/envs/foo_env.py` (loaded by `train_sac.py`).

### 1.1 Action mapping
- Action vector is 18-D in `[0,1]`:
  - `a[0] = tau` (EH phase ratio)
  - `a[1]` mapped to transmit power by
    \[
    P_1 = 10^{\left(\frac{(a_1-1)\cdot 30}{10} + 3\right)}
    \]
  - `a[2:]` are RIS phase controls, quantized to 8 levels:
    \[
    \theta_n \in \left\{0, \frac{2\pi}{8}, \ldots, \frac{14\pi}{8}\right\}
    \]

### 1.2 BS-RIS large-scale channel
- LoS probability:
  \[
  p_{\text{LoS}}=\frac{1}{1+a\exp(ab-b\theta)}
  \]
- NLoS probability: \(p_{\text{NLoS}}=1-p_{\text{LoS}}\).
- BS-RIS channel power gain:
  \[
  g_{BR}=\left(p_{\text{LoS}}+p_{\text{NLoS}}\varphi\right) d_{BR}^{-\alpha}
  \]

### 1.3 Small-scale fading
- BS-RIS Rician fading matrix:
  \[
  \mathbf{G}_{\text{small}}=\sqrt{\frac{K}{1+K}}\mathbf{1}+\sqrt{\frac{1}{1+K}}\mathbf{N}_{\text{cplx}}
  \]
- RIS-UT Rician channel:
  \[
  \mathbf{h}_{ru}=\sqrt{\frac{K}{1+K}}\,PL\,\mathbf{1}+\sqrt{\frac{1}{1+K}}\,PL\,\mathbf{n}_{\text{rayleigh}}
  \]
  where \(PL=\sqrt{\kappa (d/1)^{-\hat{\alpha}}}\).

### 1.4 Harvested energy
- Received RF power proxy:
  \[
  P_{rx}=\sum_{i,j}\|x_{ij}\|_2\,P_1
  \]
- Harvested energy:
  \[
  E_t = \tau\,\eta\,P_{rx}
  \]

### 1.5 SINR and throughput/capacity calculations
- Effective RIS phase matrix:
  \[
  \mathbf{\Phi}=\mathrm{diag}(e^{j\theta_1},\ldots,e^{j\theta_{L}})
  \]
- Cascaded link:
  \[
  \mathbf{u}=\mathbf{G}\mathbf{\Phi}\mathbf{h}_{ru}
  \]
- Signal term:
  \[
  S=\sum |\mathbf{u}|^2 P_1
  \]
- Evaluation scripts use linear SINR:
  \[
  \mathrm{SINR}_{\text{lin}}=\frac{S}{\sigma^2}
  \]
  and
  \[
  R = BW\log_2(1+\mathrm{SINR}_{\text{lin}})(1-\tau)
  \]
- Environment `capacity()` computes an internal SINR-like value in dB then applies
  \(BW\log_2(1+\cdot)(1-\tau)\), and enforces a throughput threshold; if below threshold, reward is zeroed.

### 1.6 RL reward used for training
- Raw environment reward is harvested energy (subject to throughput constraint).
- Returned step reward is normalized as EH efficiency proxy:
  \[
  r_t = \frac{\text{harvested energy}}{\text{received RF energy}}
  \]
  (with epsilon guard in denominator).

## 2) SAC setup actually used

From `train_sac.py`:
- Policy: `MlpPolicy`
- Algorithm: `stable_baselines3.SAC`
- Hyperparameters:
  - `learning_rate=3e-4`
  - `buffer_size=1_000_000`
  - `batch_size=256`
  - `seed` configurable (`--seed`, default 0)
  - `timesteps` configurable (`--timesteps`, default 200000)
  - device: auto/cuda/cpu via `--device`
- Environment for SAC script:
  - `gym.make("foo-v0", Train=True)`
  - path setup forces env import from `TD3-SingleUT-Time`.

## 3) Plot generation workflow

### 3.1 Single-model plot (`eval_plots.py`)
- Loads one model (`--algo sac|ppo|td3|ddpg`).
- Sweeps fixed Tx powers `Pt=[10,15,20,25,30] dBm`.
- For each Pt:
  - overwrite action power component (`action[1]=Pt/30.0`)
  - run episodes, compute per-step SINR and Sum-Rate from channel equations
  - average over steps and episodes
- Saves:
  - `SINR_vs_Pt.png`
  - `SumRate_vs_Pt.png`

### 3.2 Multi-model comparison (`eval_compare.py`)
- Accepts repeated `--model algo=path` specs.
- Same Pt sweep and averaging procedure.
- Saves:
  - `SINR_vs_Pt_compare.png`
  - `SumRate_vs_Pt_compare.png`

### 3.3 Rician-K sweep (`rician_k_sweep.py`)
- Trains/evaluates TD3, DDPG, SAC for K in `[0.5, 1]`.
- Uses same model hyperparameters across algorithms for fairness.
- Saves under `results/K_<K>/`:
  - `sinr_vs_pt.png`
  - `sumrate_vs_pt.png`
  - `training_reward.png`
- Also copies these plots into per-algorithm subfolders.

## 4) New SAC/modern workflow items currently present in repo

- Modern SB3 training/eval scripts exist at repo root:
  - `train_sac.py`, `train_td3.py`, `train_ddpg.py`
  - `eval_plots.py`, `eval_compare.py`, `rician_k_sweep.py`
- `run_all.sh` automates install -> static check -> train all three algos -> evaluate.
- Environment includes backward-compatible `RicianK` constructor arg and stores both `Rician_K` and `rician_k` globals for compatibility.

## 5) Fast command references you can cite in paper/repro appendix

```bash
python train_sac.py --timesteps 200000 --model-name sac_final --log-dir sac_logs
python eval_plots.py --algo sac --model-path sac_logs/sac_final.zip --episodes-per-pt 5 --output-dir eval_plots
python eval_compare.py --model sac=sac_logs/sac_final.zip --model td3=td3_logs/td3_final.zip --model ddpg=ddpg_logs/ddpg_final.zip
python rician_k_sweep.py --timesteps 200000 --episodes-per-pt 5 --seed 0 --device auto
bash run_all.sh
```
