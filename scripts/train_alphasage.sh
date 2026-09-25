#!/usr/bin/env bash
# AlphaSAGE sweep: 5 markets x 3 seeds at the official budget.
#
# Runs in .venv-alphasage — the GFlowNet stack pins numpy < 2, which the main
# environment cannot take. See factor_mining/vendor/alphasage/VENDOR.md for the
# setup command.
#
# Usage:
#   scripts/train_alphasage.sh                        # cpu, pool 50
#   DEVICE=cuda:0 scripts/train_alphasage.sh          # GPU node
#   MARKETS=sp100 SEEDS=0 N_EPISODES=200 scripts/train_alphasage.sh   # smoke
#   DRY_RUN=1 scripts/train_alphasage.sh
#
# Overridable: MARKETS, SEEDS, DEVICE, POOL_CAPACITY, N_EPISODES, ENCODER_TYPE,
# ENTROPY_COEF, SSL_WEIGHT, NOV_WEIGHT, OUT_DIR.
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cpu}"
POOL_CAPACITY="${POOL_CAPACITY:-50}"
N_EPISODES="${N_EPISODES:-10000}"
ENCODER_TYPE="${ENCODER_TYPE:-gnn}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"
SSL_WEIGHT="${SSL_WEIGHT:-1.0}"
NOV_WEIGHT="${NOV_WEIGHT:-0.3}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv-alphasage/bin/python

if [ ! -x "$PY" ]; then
  echo "[train_alphasage] missing $PY — see factor_mining/vendor/alphasage/VENDOR.md" >&2
  exit 2
fi

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.alphasage.run
         --market "$market" --seed "$seed"
         --pool_capacity "$POOL_CAPACITY" --n_episodes "$N_EPISODES"
         --encoder_type "$ENCODER_TYPE" --entropy_coef "$ENTROPY_COEF"
         --ssl_weight "$SSL_WEIGHT" --nov_weight "$NOV_WEIGHT"
         --device "$DEVICE" --out_dir "$OUT_DIR")
    echo "[train_alphasage] ${cmd[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_alphasage] FAILED: $market seed $seed (continuing)" >&2
      failed+=("$market/$seed")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_alphasage] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_alphasage] all runs completed"
