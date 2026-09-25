#!/usr/bin/env bash
# AlphaCFG sweep: 5 markets x 3 seeds at the paper-pool configuration
# (cfg-sem-k, pool mode, TreeLSTM, capacity 10, 200 iterations, MCTS-sim 64 —
# the run.py defaults). Requires dgl in .venv for the TreeLSTM backend.
#
# Usage:
#   DEVICE=cuda:0 scripts/train_alphacfg.sh
#   NETWORK=lstm MARKETS="csi300" SEEDS="0" scripts/train_alphacfg.sh
#   DRY_RUN=1 scripts/train_alphacfg.sh
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
VARIANT="${VARIANT:-cfg-sem-k}"
MODE="${MODE:-pool}"
NETWORK="${NETWORK:-treelstm}"
POOL_CAPACITY="${POOL_CAPACITY:-10}"
DEVICE="${DEVICE:-cpu}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv/bin/python

if [ "$NETWORK" = "treelstm" ] && ! "$PY" -c 'import dgl' 2>/dev/null; then
  echo "[train_alphacfg] dgl missing — run: $PY -m pip install dgl==1.1.3" >&2
  exit 1
fi

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.alphacfg.run
         --market "$market" --seed "$seed" --variant "$VARIANT" --mode "$MODE"
         --network "$NETWORK" --pool_capacity "$POOL_CAPACITY"
         --device "$DEVICE" --out_dir "$OUT_DIR")
    echo "[train_alphacfg] ${cmd[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_alphacfg] FAILED: $market seed $seed (continuing)" >&2
      failed+=("$market/$seed")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_alphacfg] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_alphacfg] all runs completed"
