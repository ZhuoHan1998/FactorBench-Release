#!/usr/bin/env bash
# quantaalpha on ftse100, seeds 0 1 2.
#
#   scripts/runs/quantaalpha_ftse100.sh
#   SEEDS="0" scripts/runs/quantaalpha_ftse100.sh     # one seed
#   DRY_RUN=1 scripts/runs/quantaalpha_ftse100.sh     # print commands only
#
# Budget and venv come from scripts/train_quantaalpha.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" quantaalpha ftse100 "$@"
