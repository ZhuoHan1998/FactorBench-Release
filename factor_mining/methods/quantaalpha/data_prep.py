"""Panel export for QuantaAlpha — identical contract to the AlphaAgent method.

QuantaAlpha inherited AlphaAgent's data contract verbatim (``daily_pv.h5``,
key ``"data"``, ``(datetime, instrument)`` index, ``$``-prefixed official
variable set with ``$return`` = 1-day close-to-close return), so this reuses
that exporter with QuantaAlpha's own cache location.
"""

from __future__ import annotations

from pathlib import Path

from factor_bench.paths import panel_derived_dir
from factor_mining.methods.alphaagent.data_prep import (
    prepare_data_folders as _prepare_alphaagent_folders,
)


def prepare_data_folders(
    market: str, base_dir: str | Path | None = None
) -> tuple[Path, Path]:
    if base_dir is None:
        base_dir = panel_derived_dir("out/mining/quantaalpha/data", "quantaalpha")
    return _prepare_alphaagent_folders(market, base_dir)
