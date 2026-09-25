#!/usr/bin/env bash
# AlphaForge sweep: 5 markets x 3 seeds at the official budget (zoo_size 100).
#
# Usage (from anywhere; runs relative to the repo root):
#   scripts/train_alphaforge.sh                      # cpu, zoo 100
#   DEVICE=cuda:0 scripts/train_alphaforge.sh        # GPU node
#   MARKETS="sp100 csi300" SEEDS="0" scripts/train_alphaforge.sh
#   DRY_RUN=1 scripts/train_alphaforge.sh            # print commands only
#   COMBINE=0 scripts/train_alphaforge.sh            # skip stage 2
#
# Stage 1 (mining) writes factors.json — that is what evaluation consumes.
# Stage 2 (dynamic combination) is AlphaForge's own combiner; it writes
# combined_signal.parquet next to it and is not part of the shared contract.
#
# Overridable env vars: MARKETS, SEEDS, DEVICE, ZOO_SIZE, MAX_ROUNDS,
# N_FACTORS, WINDOW, COMBINE, OUT_DIR.
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cpu}"
ZOO_SIZE="${ZOO_SIZE:-100}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"
N_FACTORS="${N_FACTORS:-10}"
WINDOW="${WINDOW:-inf}"
COMBINE="${COMBINE:-1}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv/bin/python

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.alphaforge.run
         --market "$market" --seed "$seed"
         --zoo_size "$ZOO_SIZE" --max_rounds "$MAX_ROUNDS"
         --device "$DEVICE" --out_dir "$OUT_DIR")
    echo "[train_alphaforge] ${cmd[*]}"
    run="$OUT_DIR/alphaforge/${market}_${ZOO_SIZE}_${seed}/factors.json"
    combine=("$PY" -m factor_mining.methods.alphaforge.combine
             --run "$run" --n_factors "$N_FACTORS" --window "$WINDOW"
             --device "$DEVICE")
    [ "$COMBINE" = "1" ] && echo "[train_alphaforge] ${combine[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_alphaforge] FAILED (mine): $market seed $seed (continuing)" >&2
      failed+=("$market/$seed:mine")
      continue
    fi
    if [ "$COMBINE" = "1" ] && ! "${combine[@]}"; then
      echo "[train_alphaforge] FAILED (combine): $market seed $seed (continuing)" >&2
      failed+=("$market/$seed:combine")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_alphaforge] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_alphaforge] all runs completed"
