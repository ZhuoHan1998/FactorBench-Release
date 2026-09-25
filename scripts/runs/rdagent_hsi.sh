#!/usr/bin/env bash
# rdagent on hsi, seeds 0 1 2.
#
#   scripts/runs/rdagent_hsi.sh
#   SEEDS="0" scripts/runs/rdagent_hsi.sh     # one seed
#   DRY_RUN=1 scripts/runs/rdagent_hsi.sh     # print commands only
#
# Budget and venv come from scripts/train_rdagent.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" rdagent hsi "$@"
