#!/usr/bin/env bash
# alphaagent on ftse100, seeds 0 1 2.
#
#   scripts/runs/alphaagent_ftse100.sh
#   SEEDS="0" scripts/runs/alphaagent_ftse100.sh     # one seed
#   DRY_RUN=1 scripts/runs/alphaagent_ftse100.sh     # print commands only
#
# Budget and venv come from scripts/train_alphaagent.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" alphaagent ftse100 "$@"
