#!/usr/bin/env bash
# alphasage on nikkei225, seeds 0 1 2.
#
#   scripts/runs/alphasage_nikkei225.sh
#   SEEDS="0" scripts/runs/alphasage_nikkei225.sh     # one seed
#   DRY_RUN=1 scripts/runs/alphasage_nikkei225.sh     # print commands only
#
# Budget and venv come from scripts/train_alphasage.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" alphasage nikkei225 "$@"
