#!/usr/bin/env bash
# quantaalpha on hsi, seeds 0 1 2.
#
#   scripts/runs/quantaalpha_hsi.sh
#   SEEDS="0" scripts/runs/quantaalpha_hsi.sh     # one seed
#   DRY_RUN=1 scripts/runs/quantaalpha_hsi.sh     # print commands only
#
# Budget and venv come from scripts/train_quantaalpha.sh; see scripts/runs/README.md.
exec "$(dirname "$0")/_driver.sh" quantaalpha hsi "$@"
