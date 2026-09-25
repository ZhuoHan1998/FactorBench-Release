#!/usr/bin/env bash
# RD-Agent sweep: 5 markets x 3 seeds, LOOPS R->D->F cycles each.
#
# Requires: .env at the repo root (LLM key/models) and both venvs installed.
# Runs are sequential on purpose — concurrent runs on the same market would
# race on the shared data cache (out/mining/rdagent/data/<market>), and each
# run already parallelizes its LLM work internally.
#
# NOTE on seeds: RD-Agent has no RNG seeding for LLM sampling. The seed only
# labels independent repetitions (and the run directory); it does not make
# runs reproducible.
#
# Usage:
#   scripts/train_rdagent.sh
#   LOOPS=10 MARKETS="csi300" SEEDS="0" scripts/train_rdagent.sh
#   DRY_RUN=1 scripts/train_rdagent.sh
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
LOOPS="${LOOPS:-5}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv-rdagent/bin/python

if [ ! -f .env ]; then
  echo "[train_rdagent] missing .env at repo root (LLM config) — aborting" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "[train_rdagent] missing $PY — run the .venv-rdagent setup first" >&2
  exit 1
fi

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.rdagent.loop
         --market "$market" --seed "$seed" --loops "$LOOPS" --out_dir "$OUT_DIR")
    echo "[train_rdagent] ${cmd[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_rdagent] FAILED: $market seed $seed (continuing)" >&2
      failed+=("$market/$seed")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_rdagent] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_rdagent] all runs completed"
