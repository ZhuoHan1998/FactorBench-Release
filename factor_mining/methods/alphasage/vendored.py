"""Import bootstrap for the vendored AlphaSAGE snapshot.

The snapshot keeps its upstream layout: the method packages live under
``src/`` (imported bare, e.g. ``alphagen.data.expression``) while the training
entry points import them as ``src.alpha_gfn.*``. Both the snapshot root and its
``src/`` directory therefore go on ``sys.path``.

AlphaSAGE ships its own fork of ``alphagen`` — a descendant of AlphaForge's,
with the operators renamed again (``TsMean`` for ``ts_mean``, ``SLog1p`` for
``S_log1p``). It is the fourth tree in this repo claiming that package name, so
``use_alphasage`` refuses rather than silently handing back another fork's
classes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from factor_mining.vendor import VENDOR_ROOT, use_vendored

_SNAPSHOT = VENDOR_ROOT / "alphasage"
_ready = False


def _assert_no_foreign_alphagen() -> None:
    """Fail loudly if another vendored fork already owns ``alphagen``."""
    module = sys.modules.get("alphagen")
    if module is None:
        return
    locations = [Path(p).resolve() for p in getattr(module, "__path__", [])]
    if any(_SNAPSHOT in loc.parents or _SNAPSHOT == loc for loc in locations):
        return
    raise RuntimeError(
        "another vendored 'alphagen' fork is already imported in this process "
        f"({locations}); AlphaSAGE's fork renames the operators and must be used "
        "in a separate process"
    )


def use_alphasage() -> None:
    """Make the vendored AlphaSAGE packages importable (idempotent)."""
    global _ready
    if _ready:
        return
    _assert_no_foreign_alphagen()
    use_vendored("alphasage")
    src = str(_SNAPSHOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    _ready = True
