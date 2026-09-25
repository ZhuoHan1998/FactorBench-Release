#!/usr/bin/env bash
# Audit factor executability, one CSV per method.
#
#   scripts/audit_factors.sh                      # all methods
#   METHODS="rdagent alphaagent" scripts/audit_factors.sh
#   LIMIT=5 scripts/audit_factors.sh              # smoke test, 5 factors per run
#   RESUME=1 scripts/audit_factors.sh             # continue a killed job
#
# On SLURM, submit per method so they run in parallel and one method's missing
# venv cannot stall the rest:
#
#   for m in gp alphagen alphacfg alphaqcm alphaforge alphasage \
#            alphaagent quantaalpha rdagent; do
#     sbatch --job-name="audit-$m" --time=04:00:00 --wrap \
#       "cd ~/workspace/FactorBench && METHODS=$m scripts/audit_factors.sh"
#   done
#
# Methods run under their own interpreter (see factor_bench/eval/executors.py).
# A method whose venv is missing reports every factor as a failure rather than
# silently using this one — the audit header prints the interpreter and pandas
# version actually used, which is the check that matters.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

PY="${PY:-.venv/bin/python}"
OUT_PREFIX="${OUT_PREFIX:-out/factor_audit}"
METHODS="${METHODS:-gp alphagen alphacfg alphaqcm alphaforge alphasage alphaagent quantaalpha rdagent}"
MINING_DIR="${MINING_DIR:-out/mining}"

if [ ! -x "$PY" ]; then
  echo "[audit] no interpreter at '$PY' (set PY=... to override)" >&2
  exit 2
fi

TAG="audit"
ok=""; bad=""
for method in $METHODS; do
  csv="${OUT_PREFIX}_${method}.csv"
  args=(--methods "$method" --out_csv "$csv" --mining_dir "$MINING_DIR")
  [ -n "${LIMIT:-}" ]  && args+=(--limit "$LIMIT")
  [ -n "${RESUME:-}" ] && args+=(--resume)

  echo "[$TAG] === $method -> $csv ==="
  if "$PY" -m factor_bench.eval.audit "${args[@]}"; then
    ok="$ok $method"
  else
    echo "[$TAG] FAILED: $method (audit itself errored; see output above)" >&2
    bad="$bad $method"
  fi
  echo
done

echo "[$TAG] completed:$ok"
if [ -n "$bad" ]; then
  echo "[$TAG] audit errors:$bad" >&2
  exit 1
fi

echo "[$TAG] combine with:"
echo "  $PY -c \"import pandas as pd,glob; d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob('${OUT_PREFIX}_*.csv'))]); d.to_csv('${OUT_PREFIX}_all.csv',index=False); from factor_bench.eval.audit import print_summary; print_summary(d)\""
