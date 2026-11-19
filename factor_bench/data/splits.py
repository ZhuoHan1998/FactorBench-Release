"""Canonical chronological split shared by all mining methods and evaluation.

The split is defined once here and imported everywhere. Mining methods train
on ``train``, may use ``valid`` for model selection / early stopping, and must
never see ``test``. Evaluation reports metrics per segment so that search
overfitting (train IC high, valid IC collapsed) is measurable.

Buffer rationale (do not tighten without checking both ends):
- Cached panels start 2016-01-04. Train starts 2017-01-01, leaving ~250
  trading days of warm-up history before the first train date — enough for
  the largest lookback any method uses (AlphaGen's ``max_backtrack_days=100``
  plus rolling-window operators up to 40 days).
- Cached panels end 2025-12-30. Test ends 2025-10-31, leaving >30 trading
  days so the 20-day forward-return target (and AlphaGen's
  ``max_future_days=30``) is fully defined on every test date.

Purging between segments: the target at date t is the 20-trading-day forward
return, so a label dated in the last 20 trading days of a segment is realised
inside the next one. Train and valid therefore end early: 2021-11-30 and
2023-11-29 are the latest dates whose labels end before the next segment
starts in every market (FTSE 100 and HSI bind; checked on each market's own
trading calendar). The trading days in between are deliberately unused.
Without this, anything selected on valid would have seen roughly the first
month of test returns. No embargo is needed: segments only run forward in
time, so no earlier segment is ever fitted after a later one.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    """Chronological train/valid/test split. All bounds are inclusive dates."""

    train: tuple[str, str]
    valid: tuple[str, str]
    test: tuple[str, str]

    def __post_init__(self) -> None:
        segments = [self.train, self.valid, self.test]
        bounds = [pd.Timestamp(d) for seg in segments for d in seg]
        if bounds != sorted(bounds):
            raise ValueError(f"Split segments must be chronological and non-overlapping: {self}")

    def segment(self, name: str) -> tuple[str, str]:
        if name not in ("train", "valid", "test"):
            raise ValueError(f"Unknown segment {name!r}")
        return getattr(self, name)

    def mask(self, dates: pd.Index, segment: str) -> pd.Index:
        """Boolean mask over a date index selecting one segment (inclusive)."""
        start, end = self.segment(segment)
        return (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))

    def as_dict(self) -> dict[str, tuple[str, str]]:
        return {"train": self.train, "valid": self.valid, "test": self.test}


DEFAULT_SPLIT = Split(
    train=("2017-01-01", "2021-11-30"),   # purged: last label ends before 2022-01-01
    valid=("2022-01-01", "2023-11-29"),   # purged: last label ends before 2024-01-01
    test=("2024-01-01", "2025-10-31"),
)
