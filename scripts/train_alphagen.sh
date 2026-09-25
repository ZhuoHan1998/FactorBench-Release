#!/usr/bin/env bash
# AlphaGen sweep: 5 markets x 3 seeds at the official budget.
#
# Usage (from anywhere; runs relative to the repo root):
#   scripts/train_alphagen.sh                       # cpu, capacity 10
#   DEVICE=cuda:0 scripts/train_alphagen.sh         # GPU node
#   MARKETS="sp100 csi300" SEEDS="0" scripts/train_alphagen.sh
#   DRY_RUN=1 scripts/train_alphagen.sh             # print commands only
#
# Overridable env vars: MARKETS, SEEDS, DEVICE, POOL_CAPACITY, STEPS (optional
# override of the official budget), OUT_DIR.
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cpu}"
POOL_CAPACITY="${POOL_CAPACITY:-10}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv/bin/python

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.alphagen.run
         --market "$market" --seed "$seed"
         --pool_capacity "$POOL_CAPACITY" --device "$DEVICE" --out_dir "$OUT_DIR")
    [ -n "${STEPS:-}" ] && cmd+=(--steps "$STEPS")
    echo "[train_alphagen] ${cmd[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_alphagen] FAILED: $market seed $seed (continuing)" >&2
      failed+=("$market/$seed")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_alphagen] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_alphagen] all runs completed"
