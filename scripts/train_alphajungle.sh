#!/usr/bin/env bash
# Alpha Jungle sweep: 5 markets x 3 seeds at the paper's smallest node budget.
#
# NOTE: Alpha Jungle has no official implementation; this is a from-paper
# reimplementation. See factor_mining/methods/alphajungle/METHOD.md for the
# assumptions it rests on before citing any number from it.
#
# Requires an LLM endpoint, configured the same way as the other LLM methods:
#   export OPENAI_API_BASE=...   # or OPENAI_BASE_URL
#   export OPENAI_API_KEY=...
#   export CHAT_MODEL=...
#
# Usage:
#   scripts/train_alphajungle.sh
#   MARKETS="sp100" SEEDS="0" NUM_NODES=50 scripts/train_alphajungle.sh   # smoke
#   DRY_RUN=1 scripts/train_alphajungle.sh
#
# Cost: 3.75 LLM calls per node measured, so the default 1000-node budget is
# ~3.75k calls per seed and ~56k for the full 15-run sweep. An expansion is a
# strict LLM chain, so nothing inside a tree overlaps; CONCURRENCY (default 8)
# grows that many independent trees at once. See METHOD.md.
#
# Overridable env vars: MARKETS, SEEDS, NUM_NODES, TOP_K, CONCURRENCY, MODEL,
# OUT_DIR.
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
NUM_NODES="${NUM_NODES:-1000}"
CONCURRENCY="${CONCURRENCY:-8}"
TOP_K="${TOP_K:-10}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv/bin/python

if [ ! -f .env ] && [ -z "${OPENAI_API_BASE:-}${OPENAI_BASE_URL:-}" ]; then
  echo "[train_alphajungle] set OPENAI_API_BASE (or provide a repo-root .env)" >&2
  exit 2
fi
if [ ! -f .env ] && [ -z "${CHAT_MODEL:-}${MODEL:-}" ]; then
  echo "[train_alphajungle] set CHAT_MODEL (or provide a repo-root .env)" >&2
  exit 2
fi

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.alphajungle.run
         --market "$market" --seed "$seed"
         --num_nodes "$NUM_NODES" --top_k "$TOP_K"
         --concurrency "$CONCURRENCY" --out_dir "$OUT_DIR")
    [ -n "${MODEL:-}" ] && cmd+=(--model "$MODEL")
    echo "[train_alphajungle] ${cmd[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_alphajungle] FAILED: $market seed $seed (continuing)" >&2
      failed+=("$market/$seed")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_alphajungle] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_alphajungle] all runs completed"
