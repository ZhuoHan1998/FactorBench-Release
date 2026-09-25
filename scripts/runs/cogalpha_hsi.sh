#!/usr/bin/env bash
# cogalpha on hsi, seeds 0 1 2.
#
#   scripts/runs/cogalpha_hsi.sh
#   SEEDS="0" scripts/runs/cogalpha_hsi.sh     # one seed
#   DRY_RUN=1 scripts/runs/cogalpha_hsi.sh     # print commands only
#
# Budget and venv come from scripts/train_cogalpha.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" cogalpha hsi "$@"
