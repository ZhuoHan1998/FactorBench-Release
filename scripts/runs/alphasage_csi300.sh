#!/usr/bin/env bash
# alphasage on csi300, seeds 0 1 2.
#
#   scripts/runs/alphasage_csi300.sh
#   SEEDS="0" scripts/runs/alphasage_csi300.sh     # one seed
#   DRY_RUN=1 scripts/runs/alphasage_csi300.sh     # print commands only
#
# Budget and venv come from scripts/train_alphasage.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" alphasage csi300 "$@"
