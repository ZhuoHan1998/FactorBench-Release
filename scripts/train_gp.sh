#!/usr/bin/env bash
# GP baseline sweep: 5 markets x 3 seeds at the official budget.
#
# This is AlphaGen's own genetic-programming baseline (its gp.py), vendored
# with the rest of AlphaGen. See factor_mining/methods/gp/run.py for the three
# export gates that keep unevaluable programs out of factors.json.
#
# Usage:
#   scripts/train_gp.sh                                  # cpu, pool 20
#   MARKETS=sp100 SEEDS=0 GENERATIONS=6 POPULATION=200 scripts/train_gp.sh
#   DRY_RUN=1 scripts/train_gp.sh
#
# Overridable: MARKETS, SEEDS, POOL_CAPACITY, POPULATION, GENERATIONS,
# TOURNAMENT, MUTUAL_IC_THRES, DEVICE, OUT_DIR.
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
POOL_CAPACITY="${POOL_CAPACITY:-20}"
POPULATION="${POPULATION:-1000}"
GENERATIONS="${GENERATIONS:-40}"
TOURNAMENT="${TOURNAMENT:-600}"
MUTUAL_IC_THRES="${MUTUAL_IC_THRES:-0.7}"
DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv/bin/python

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.gp.run
         --market "$market" --seed "$seed"
         --pool_capacity "$POOL_CAPACITY" --population_size "$POPULATION"
         --generations "$GENERATIONS" --tournament_size "$TOURNAMENT"
         --mutual_ic_thres "$MUTUAL_IC_THRES"
         --device "$DEVICE" --out_dir "$OUT_DIR")
    echo "[train_gp] ${cmd[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_gp] FAILED: $market seed $seed (continuing)" >&2
      failed+=("$market/$seed")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_gp] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_gp] all runs completed"
