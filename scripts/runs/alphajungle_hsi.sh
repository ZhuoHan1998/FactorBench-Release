#!/usr/bin/env bash
# alphajungle on hsi, seeds 0 1 2.
#
#   scripts/runs/alphajungle_hsi.sh
#   SEEDS="0" scripts/runs/alphajungle_hsi.sh     # one seed
#   DRY_RUN=1 scripts/runs/alphajungle_hsi.sh     # print commands only
#
# Budget and venv come from scripts/train_alphajungle.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" alphajungle hsi "$@"
