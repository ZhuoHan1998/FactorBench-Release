"""Vendored official implementations, kept unmodified.

Each subdirectory is a pinned snapshot of an official method repository (see
its VENDOR.md for provenance). Snapshots keep their original top-level package
names, so they are put on sys.path explicitly rather than imported as
submodules of this package.
"""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent


# Several snapshots ship their own fork of the same top-level packages (four
# provide `alphagen`, with different operator names and semantics). Whichever
# is imported first wins for the whole process, so a conflict must fail loudly
# rather than silently handing one fork's classes to code expecting another's.
_SHARED_PACKAGE_NAMES = ("alphagen", "alphagen_qlib", "alphagen_generic")


def _assert_no_conflicting_fork(snapshot: Path) -> None:
    for package in _SHARED_PACKAGE_NAMES:
        module = sys.modules.get(package)
        if module is None:
            continue
        locations = [Path(p).resolve() for p in getattr(module, "__path__", [])]
        if not locations or any(
            snapshot == loc or snapshot in loc.parents for loc in locations
        ):
            continue
        raise RuntimeError(
            f"'{package}' is already imported in this process from {locations[0]}, "
            f"which is a different vendored fork than {snapshot}. These forks "
            "rename operators and are not interchangeable — use a separate "
            "process (see factor_bench.eval.executors)."
        )


def use_vendored(name: str) -> None:
    """Make a vendored snapshot's top-level packages importable (idempotent)."""
    path = VENDOR_ROOT / name
    if not path.is_dir():
        raise FileNotFoundError(f"No vendored snapshot at {path}")
    _assert_no_conflicting_fork(path)
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)
