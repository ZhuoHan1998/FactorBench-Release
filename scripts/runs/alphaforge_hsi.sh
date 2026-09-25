#!/usr/bin/env bash
# alphaforge on hsi, seeds 0 1 2.
#
#   scripts/runs/alphaforge_hsi.sh
#   SEEDS="0" scripts/runs/alphaforge_hsi.sh     # one seed
#   DRY_RUN=1 scripts/runs/alphaforge_hsi.sh     # print commands only
#
# Budget and venv come from scripts/train_alphaforge.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" alphaforge hsi "$@"
