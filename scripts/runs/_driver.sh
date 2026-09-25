#!/usr/bin/env bash
# Shared driver for the per-(method, market) job scripts in this directory.
#
#   scripts/runs/_driver.sh <method> <market>
#
# Not meant to be called directly — use the generated wrappers, e.g.
# scripts/runs/gp_csi300.sh. They exist so each (method, market) pair is one
# schedulable unit: submit them individually, run them in parallel, retry one
# without touching the others.
#
# This delegates to scripts/train_<method>.sh, which owns each method's venv,
# budget defaults and prerequisites. Nothing about a method is duplicated here;
# this only pins the market, fixes the seed list, checks prerequisites, logs,
# and reports what landed on disk.
#
# Overridable: SEEDS (default "0 1 2"), OUT_DIR, LOG_DIR, DRY_RUN, plus any env
# var the underlying train_*.sh reads (DEVICE, POOL_CAPACITY, ...).
set -uo pipefail
cd "$(dirname "$0")/../.."

METHOD="${1:-}"
MARKET="${2:-}"
if [ -z "$METHOD" ] || [ -z "$MARKET" ]; then
  echo "usage: scripts/runs/_driver.sh <method> <market>" >&2
  exit 2
fi

SEEDS="${SEEDS:-0 1 2}"
OUT_DIR="${OUT_DIR:-out/mining}"
LOG_DIR="${LOG_DIR:-out/logs/$MARKET}"
SCRIPT="scripts/train_${METHOD}.sh"

case "$METHOD" in
  alphasage)   VENV=".venv-alphasage" ;;
  alphaagent)  VENV=".venv-alphaagent" ;;
  quantaalpha) VENV=".venv-quantaalpha" ;;
  rdagent)     VENV=".venv-rdagent" ;;
  *)           VENV=".venv" ;;
esac
case "$METHOD" in
  alphajungle|alphaagent|rdagent|quantaalpha|cogalpha) NEEDS_LLM=1 ;;
  *) NEEDS_LLM=0 ;;
esac

if [ ! -x "$SCRIPT" ]; then
  echo "[$METHOD/$MARKET] no $SCRIPT — nothing to run" >&2
  exit 2
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "[$METHOD/$MARKET] missing $VENV — see the method's VENDOR.md for setup" >&2
  exit 2
fi
if [ "$NEEDS_LLM" = 1 ] && [ ! -f .env ] && [ -z "${OPENAI_API_BASE:-}${OPENAI_BASE_URL:-}" ]; then
  echo "[$METHOD/$MARKET] needs LLM config (.env at the repo root, or OPENAI_API_BASE)" >&2
  exit 2
fi
# `MODEL` means the algorithm variant to AlphaQCM but the LLM name to the LLM
# methods, so a globally exported value misconfigures one of them.
if [ -n "${MODEL:-}" ] && [ "$METHOD" = "alphaqcm" ]; then
  echo "[$METHOD/$MARKET] note: MODEL='$MODEL' is read as the AlphaQCM algorithm" >&2
  echo "[$METHOD/$MARKET]       variant (iqn/qrdqn), not an LLM name." >&2
fi

if [ -n "${DRY_RUN:-}" ]; then
  DRY_RUN=1 MARKETS="$MARKET" SEEDS="$SEEDS" OUT_DIR="$OUT_DIR" "$SCRIPT"
  exit 0
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${METHOD}.log"
echo "[$METHOD/$MARKET] seeds '$SEEDS' -> $LOG"
t0=$(date +%s)
MARKETS="$MARKET" SEEDS="$SEEDS" OUT_DIR="$OUT_DIR" "$SCRIPT" >"$LOG" 2>&1
status=$?
mins=$(( ($(date +%s) - t0) / 60 ))

echo "[$METHOD/$MARKET] finished in ${mins}m (exit $status)"
found=0
for f in "$OUT_DIR/$METHOD/${MARKET}"_*/factors.json; do
  [ -f "$f" ] || continue
  found=1
  n=$(.venv/bin/python -c "import json,sys;print(len(json.load(open(sys.argv[1]))['factors']))" "$f" 2>/dev/null || echo "?")
  echo "[$METHOD/$MARKET]   $(dirname "$f")  ($n factors)"
done
[ "$found" = 0 ] && echo "[$METHOD/$MARKET]   no factors.json produced"
[ "$status" -ne 0 ] && tail -5 "$LOG" | sed 's/^/    | /'
exit $status
