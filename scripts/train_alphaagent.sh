#!/usr/bin/env bash
# AlphaAgent sweep: 5 markets x 3 seeds, ROUNDS agent rounds each.
#
# Requires: .env at the repo root (LLM config incl. REASONING_MODEL) and the
# .venv-alphaagent virtualenv. Runs are sequential: each run makes many LLM
# calls, and concurrent runs on one market would race on the shared data
# cache (out/mining/alphaagent/data/<market>).
#
# NOTE on seeds: AlphaAgent has no RNG seeding for LLM sampling. The seed only
# labels independent repetitions; it does not make runs reproducible.
#
# Usage:
#   scripts/train_alphaagent.sh
#   ROUNDS=5 MARKETS="csi300" SEEDS="0" scripts/train_alphaagent.sh
#   DRY_RUN=1 scripts/train_alphaagent.sh
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
ROUNDS="${ROUNDS:-3}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv-alphaagent/bin/python

if [ ! -f .env ]; then
  echo "[train_alphaagent] missing .env at repo root (LLM config) — aborting" >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "[train_alphaagent] missing $PY — run the .venv-alphaagent setup first" >&2
  exit 1
fi

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.alphaagent.run
         --market "$market" --seed "$seed" --rounds "$ROUNDS" --out_dir "$OUT_DIR")
    echo "[train_alphaagent] ${cmd[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_alphaagent] FAILED: $market seed $seed (continuing)" >&2
      failed+=("$market/$seed")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_alphaagent] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_alphaagent] all runs completed"
