#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash run_all.sh                  # full run (200k each algo)
#   TIMESTEPS=50000 bash run_all.sh  # custom full run length
#   QUICK_TEST=1 bash run_all.sh     # smoke test mode (very fast)

TIMESTEPS="${TIMESTEPS:-200000}"
EPISODES_PER_PT="${EPISODES_PER_PT:-5}"
K_VALUES="${K_VALUES:-0.5 5.0}"
QUICK_TEST="${QUICK_TEST:-0}"

if [[ "$QUICK_TEST" == "1" ]]; then
  TIMESTEPS=5
  EPISODES_PER_PT=1
fi

echo "[1/7] Installing dependencies (set SKIP_INSTALL=1 to skip)"
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  python -m pip install -r requirements.txt || echo "[WARN] pip install failed; continuing with existing environment."
else
  echo "[INFO] Skipping dependency installation."
fi

echo "[2/7] Static check"
python -m py_compile train_sac.py train_td3.py train_ddpg.py eval_sweep.py

echo "[3/7] Train SAC (timesteps=${TIMESTEPS})"
python train_sac.py --timesteps "${TIMESTEPS}" --model-name sac_final --log-dir sac_logs

echo "[4/7] Train TD3 (timesteps=${TIMESTEPS})"
python train_td3.py --timesteps "${TIMESTEPS}" --model-name td3_final --log-dir td3_logs

echo "[5/7] Train DDPG (timesteps=${TIMESTEPS})"
python train_ddpg.py --timesteps "${TIMESTEPS}" --model-name ddpg_final --log-dir ddpg_logs

echo "[6/7] Evaluate all algorithms (K=${K_VALUES}, episodes=${EPISODES_PER_PT})"
# shellcheck disable=SC2086
python eval_sweep.py --algo all --k-values ${K_VALUES} --episodes-per-pt "${EPISODES_PER_PT}"

echo "[7/7] Done"
echo "Generated plots:"
ls -1 SINR_vs_Pt_*_Kcompare.png SumRate_vs_Pt_*_Kcompare.png SINR_SumRate_vs_Pt_*_Kcompare.png
