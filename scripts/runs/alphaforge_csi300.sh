#!/usr/bin/env bash
# alphaforge on csi300, seeds 0 1 2.
#
#   scripts/runs/alphaforge_csi300.sh
#   SEEDS="0" scripts/runs/alphaforge_csi300.sh     # one seed
#   DRY_RUN=1 scripts/runs/alphaforge_csi300.sh     # print commands only
#
# Budget and venv come from scripts/train_alphaforge.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" alphaforge csi300 "$@"
