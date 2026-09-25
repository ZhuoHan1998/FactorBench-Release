#!/usr/bin/env bash
# QuantaAlpha sweep: 5 markets x 3 seeds at the paper's hyperparameters.
#
# Requires: .env at the repo root (LLM config) and the .venv-quantaalpha
# virtualenv. Runs are sequential: each makes many LLM calls, and concurrent
# runs on one market would race on the shared data cache
# (out/mining/quantaalpha/data/<market>).
#
# NOTE on seeds: like the other LLM methods here, seeding does not make runs
# reproducible — it only labels independent repetitions.
#
# Budget defaults are the paper's (docs/experiment_hyperparameters.md:
# num_directions=10, max_rounds=11, crossover_size=2, crossover_n=10), which is
# a very large LLM budget. Override for smoke tests.
#
# Usage:
#   scripts/train_quantaalpha.sh
#   MARKETS="csi300" SEEDS="0" NUM_DIRECTIONS=2 MAX_ROUNDS=1 scripts/train_quantaalpha.sh
#   DRY_RUN=1 scripts/train_quantaalpha.sh
#
# Overridable: MARKETS, SEEDS, DIRECTION, NUM_DIRECTIONS, MAX_ROUNDS,
# CROSSOVER_SIZE, CROSSOVER_N, OUT_DIR.
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
DIRECTION="${DIRECTION:-daily price-volume alpha factors}"
NUM_DIRECTIONS="${NUM_DIRECTIONS:-10}"
MAX_ROUNDS="${MAX_ROUNDS:-11}"
CROSSOVER_SIZE="${CROSSOVER_SIZE:-2}"
CROSSOVER_N="${CROSSOVER_N:-10}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv-quantaalpha/bin/python

if [ ! -f .env ]; then
  echo "[train_quantaalpha] missing .env at repo root (LLM config) — aborting" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "[train_quantaalpha] missing $PY — run the .venv-quantaalpha setup first" >&2
  exit 1
fi

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.quantaalpha.run
         --market "$market" --seed "$seed"
         --direction "$DIRECTION"
         --num_directions "$NUM_DIRECTIONS" --max_rounds "$MAX_ROUNDS"
         --crossover_size "$CROSSOVER_SIZE" --crossover_n "$CROSSOVER_N"
         --out_dir "$OUT_DIR")
    echo "[train_quantaalpha] ${cmd[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_quantaalpha] FAILED: $market seed $seed (continuing)" >&2
      failed+=("$market/$seed")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_quantaalpha] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_quantaalpha] all runs completed"
