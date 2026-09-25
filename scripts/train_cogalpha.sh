#!/usr/bin/env bash
# CogAlpha sweep: 5 markets x 3 seeds, full 21-agent hierarchy at reduced depth.
#
# NOTE: CogAlpha has no official implementation; this is a from-paper
# reimplementation. See factor_mining/methods/cogalpha/METHOD.md before citing
# any number from it — in particular the evolution depth is 4 generations per
# agent, not the paper's 24 (which is ~500 GPU-hours per market).
#
# Requires an LLM endpoint (repo-root .env or exported):
#   OPENAI_API_BASE, OPENAI_API_KEY, CHAT_MODEL
#
# Usage:
#   scripts/train_cogalpha.sh
#   MARKETS=sp100 SEEDS=0 GENERATIONS=1 INITIAL_AGENTS=3 MAX_LLM_CALLS=60 \
#     scripts/train_cogalpha.sh        # smoke
#   DRY_RUN=1 scripts/train_cogalpha.sh
#
# Overridable: MARKETS, SEEDS, GENERATIONS, TOP_K, PARENT_POOL, INITIAL_AGENTS,
# JUDGE, LEAKAGE_TEST, MAX_LLM_CALLS, CONCURRENCY, NUM_PER_REQUEST, MODEL, OUT_DIR.
#
# Cost: ~10k LLM calls per seed at the default, 82% of them the Judge agent.
# CONCURRENCY (default 8) issues independent requests in parallel; JUDGE=False
# removes that 82% at some loss of paper fidelity. METHOD.md has the measured
# budget for each setting.
set -uo pipefail
cd "$(dirname "$0")/.."

MARKETS="${MARKETS:-sp100 hsi csi300 ftse100 nikkei225}"
SEEDS="${SEEDS:-0 1 2}"
GENERATIONS="${GENERATIONS:-4}"
TOP_K="${TOP_K:-10}"
PARENT_POOL="${PARENT_POOL:-32}"
INITIAL_AGENTS="${INITIAL_AGENTS:-13}"
JUDGE="${JUDGE:-True}"
LEAKAGE_TEST="${LEAKAGE_TEST:-True}"
MAX_LLM_CALLS="${MAX_LLM_CALLS:-0}"
CONCURRENCY="${CONCURRENCY:-8}"
NUM_PER_REQUEST="${NUM_PER_REQUEST:-6}"
OUT_DIR="${OUT_DIR:-out/mining}"
PY=.venv/bin/python

if [ ! -f .env ] && [ -z "${OPENAI_API_BASE:-}${OPENAI_BASE_URL:-}" ]; then
  echo "[train_cogalpha] set OPENAI_API_BASE (or provide a repo-root .env)" >&2
  exit 2
fi

failed=()
for market in $MARKETS; do
  for seed in $SEEDS; do
    cmd=("$PY" -m factor_mining.methods.cogalpha.run
         --market "$market" --seed "$seed"
         --generations "$GENERATIONS" --top_k "$TOP_K"
         --parent_pool "$PARENT_POOL" --initial_agents "$INITIAL_AGENTS"
         --judge "$JUDGE" --leakage_test "$LEAKAGE_TEST"
         --max_llm_calls "$MAX_LLM_CALLS" --concurrency "$CONCURRENCY"
         --num_per_request "$NUM_PER_REQUEST" --out_dir "$OUT_DIR")
    [ -n "${MODEL:-}" ] && cmd+=(--model "$MODEL")
    echo "[train_cogalpha] ${cmd[*]}"
    [ -n "${DRY_RUN:-}" ] && continue
    if ! "${cmd[@]}"; then
      echo "[train_cogalpha] FAILED: $market seed $seed (continuing)" >&2
      failed+=("$market/$seed")
    fi
  done
done

if [ "${#failed[@]}" -gt 0 ]; then
  echo "[train_cogalpha] failed runs: ${failed[*]}" >&2
  exit 1
fi
echo "[train_cogalpha] all runs completed"
