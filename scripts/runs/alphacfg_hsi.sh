#!/usr/bin/env bash
# alphacfg on hsi, seeds 0 1 2.
#
#   scripts/runs/alphacfg_hsi.sh
#   SEEDS="0" scripts/runs/alphacfg_hsi.sh     # one seed
#   DRY_RUN=1 scripts/runs/alphacfg_hsi.sh     # print commands only
#
# Budget and venv come from scripts/train_alphacfg.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" alphacfg hsi "$@"
