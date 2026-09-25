"""Panel export for QuantaAlpha — identical contract to the AlphaAgent method.

QuantaAlpha inherited AlphaAgent's data contract verbatim (``daily_pv.h5``,
key ``"data"``, ``(datetime, instrument)`` index, ``$``-prefixed official
variable set with ``$return`` = 1-day close-to-close return), so this reuses
that exporter with QuantaAlpha's own cache location.
"""

from __future__ import annotations

from pathlib import Path

from factor_mining.methods.alphaagent.data_prep import (
    prepare_data_folders as _prepare_alphaagent_folders,
)


def prepare_data_folders(
    market: str, base_dir: str | Path = "out/mining/quantaalpha/data"
) -> tuple[Path, Path]:
    return _prepare_alphaagent_folders(market, base_dir)
