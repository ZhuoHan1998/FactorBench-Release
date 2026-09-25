#!/usr/bin/env bash
# AlphaQCM sweep: 5 markets x 3 seeds at the official budget (2M steps).
#
# Usage:
#   DEVICE=cuda:0 scripts/train_alphaqcm.sh
#   MODEL=qrdqn MARKETS="csi300" SEEDS="0" scripts/train_alphaqcm.sh
#   DRY_RUN=1 scripts/train_alphaqcm.sh
#
# MODEL: iqn (used in our runs so far) or qrdqn (official script default);
# the paper's grid treats both as main variants. NUM_STEPS overrides the
# official 2M-step budget (smoke tests only).
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
MODEL="${MODEL:-iqn}"
POOL_CAPACITY="${POOL_CAPACITY:-10}"
STD_LAM="${STD_LAM:-1.0}"
DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv/bin/python

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.alphaqcm.run
         --market "$market" --seed "$seed" --model "$MODEL"
         --pool_capacity "$POOL_CAPACITY" --std_lam "$STD_LAM"
         --device "$DEVICE" --out_dir "$OUT_DIR")
    [ -n "${NUM_STEPS:-}" ] && cmd+=(--num_steps "$NUM_STEPS")
    echo "[train_alphaqcm] ${cmd[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_alphaqcm] FAILED: $market seed $seed (continuing)" >&2
      failed+=("$market/$seed")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_alphaqcm] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_alphaqcm] all runs completed"
