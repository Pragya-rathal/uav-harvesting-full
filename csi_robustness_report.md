# CSI Robustness Report

## Supported CSI modes

- `csi_mode="perfect"`
- `csi_mode="imperfect"`

## Imperfect CSI model

The environment perturbs the simulated channel estimate:

`H_est = H_true + noise`

The perturbation is Gaussian and controlled by `sigma_csi`.

For complex-valued links, the code adds independent real and imaginary noise terms.

## Decision path

- The policy and reward path consume the perturbed CSI when `csi_mode="imperfect"`.
- `sigma_csi` is configurable from the evaluation / publication suite.

## Compatibility guarantee

- Perfect CSI remains the default behavior.
- No wireless-channel equation is rewritten.
- Imperfect CSI is additive and opt-in.

## Validation placeholder

After completing training, compare:
- perfect CSI vs imperfect CSI
- reward
- harvested energy
- SINR / sum-rate
- runtime impact
