#!/usr/bin/env bash
# cogalpha on sp100, seeds 0 1 2.
#
#   scripts/runs/cogalpha_sp100.sh
#   SEEDS="0" scripts/runs/cogalpha_sp100.sh     # one seed
#   DRY_RUN=1 scripts/runs/cogalpha_sp100.sh     # print commands only
#
# Budget and venv come from scripts/train_cogalpha.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" cogalpha sp100 "$@"
