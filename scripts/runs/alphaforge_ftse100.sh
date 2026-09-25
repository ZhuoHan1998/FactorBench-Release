#!/usr/bin/env bash
# alphaforge on ftse100, seeds 0 1 2.
#
#   scripts/runs/alphaforge_ftse100.sh
#   SEEDS="0" scripts/runs/alphaforge_ftse100.sh     # one seed
#   DRY_RUN=1 scripts/runs/alphaforge_ftse100.sh     # print commands only
#
# Budget and venv come from scripts/train_alphaforge.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" alphaforge ftse100 "$@"
