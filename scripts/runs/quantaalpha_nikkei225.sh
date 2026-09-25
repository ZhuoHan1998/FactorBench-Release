#!/usr/bin/env bash
# quantaalpha on nikkei225, seeds 0 1 2.
#
#   scripts/runs/quantaalpha_nikkei225.sh
#   SEEDS="0" scripts/runs/quantaalpha_nikkei225.sh     # one seed
#   DRY_RUN=1 scripts/runs/quantaalpha_nikkei225.sh     # print commands only
#
# Budget and venv come from scripts/train_quantaalpha.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" quantaalpha nikkei225 "$@"
