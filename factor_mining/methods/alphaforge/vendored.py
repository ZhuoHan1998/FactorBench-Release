"""Import bootstrap for the vendored AlphaForge snapshot.

Two things must be true before any ``alphagen.*`` / ``gan.*`` import resolves
against the AlphaForge tree:

1. The snapshot is on ``sys.path``. AlphaForge ships its own fork of the
   ``alphagen`` / ``alphagen_qlib`` / ``alphagen_generic`` packages whose
   operator set (``ts_mean`` instead of ``Mean``, plus ``Inv``, ``S_log1p``,
   ``ts_div``, ``ts_pctchange``) and expression string form (infix, bare
   feature names) differ from upstream AlphaGen's. Two vendored trees claiming
   the same top-level package must never be imported into one process, so
   ``use_alphaforge`` refuses rather than silently handing back the other
   fork's classes.
2. ``gym`` resolves. ``alphagen/rl/env/{core,wrapper}.py`` do ``import gym``
   and upstream pins the abandoned gym 0.21, which does not build on Python
   3.11. AlphaForge's stage-1 search never constructs an environment — the
   modules are pulled in only for ``SIZE_ACTION`` / ``action2token`` / the
   token offsets — and every gym name touched at import time (``Env``,
   ``Wrapper``, ``spaces.Discrete``, ``spaces.Box``) exists in gymnasium with
   the same signature. Aliasing gymnasium into ``sys.modules`` keeps the
   vendored tree byte-identical to upstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

from factor_mining.vendor import VENDOR_ROOT, use_vendored

_SNAPSHOT = VENDOR_ROOT / "alphaforge"
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
        f"({locations}); AlphaForge's fork has different operator semantics and "
        "must be used in a separate process"
    )


def use_alphaforge() -> None:
    """Make the vendored AlphaForge packages importable (idempotent)."""
    global _ready
    if _ready:
        return
    _assert_no_foreign_alphagen()
    import gymnasium

    sys.modules.setdefault("gym", gymnasium)
    sys.modules.setdefault("gym.spaces", gymnasium.spaces)
    use_vendored("alphaforge")
    _ready = True
