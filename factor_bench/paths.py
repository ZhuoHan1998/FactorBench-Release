"""Where evaluation artefacts are written and read.

One definition, so the writers (``factor_bench.eval.*``, ``factor_bench.portfolio.*``)
and the readers (the RQ notebooks) can never drift apart. Before this existed the
path ``out/eval`` was spelled out in nine places, and ``ic_table`` wrote to
``out/`` while every notebook read from ``out/eval/``.

Why an environment variable rather than a constant
--------------------------------------------------
The artefacts are large (~60 MB of correlation matrices, plus the portfolio
CSVs) and on a cluster they usually belong on scratch, not in the repo. But an
absolute cluster path committed here would break every other machine. So the
root is read from ``FACTORBENCH_OUT`` and falls back to the in-repo ``out/``,
which keeps a fresh clone working with no setup.

Two ways to point at it, in order of precedence::

    export FACTORBENCH_OUT=/scratch/users/<user>/factorbench/out   # per shell

    echo /scratch/users/<user>/factorbench/out > .factorbench_out  # per machine

The file is the one to prefer for notebook work: a Jupyter kernel started from a
launcher or an IDE does not inherit a shell export, and a notebook that quietly
reads the wrong tree is how stale results get into figures.

Then every stage writes under ``$FACTORBENCH_OUT/eval/`` and every notebook
reads from the same place. Unset it and everything returns to ``out/eval``.

Figures are deliberately NOT covered: see :data:`FIGURE_DIR`.

Note the signal cache is NOT covered by this: it is ~10 GB of intermediate
data, it is passed explicitly with ``--signal_cache``, and it should live
wherever there is space.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Per-machine override file, sitting next to the repo root and NOT committed.
#: The environment variable is easy to forget when launching Jupyter -- and a
#: notebook that silently reads a stale tree is the failure this benchmark has
#: hit most often -- so a machine can also record its path once, here.
_CONFIG = Path(__file__).resolve().parents[1] / ".factorbench_out"


def _resolve_root() -> tuple[Path, str]:
    """(root, where it came from). Environment first, then the config file."""
    env = os.environ.get("FACTORBENCH_OUT")
    if env:
        return Path(env), "FACTORBENCH_OUT"
    if _CONFIG.exists():
        text = _CONFIG.read_text().strip()
        if text:
            return Path(text), str(_CONFIG.name)
    return Path("out"), "default"


OUT_ROOT, _ROOT_SOURCE = _resolve_root()

#: Evaluation artefacts: the IC table, and one subdirectory per stage.
EVAL_DIR = OUT_ROOT / "eval"

IC_TABLE = EVAL_DIR / "ic_table.csv"
REDUNDANCY_DIR = EVAL_DIR / "redundancy"
STYLE_DIR = EVAL_DIR / "style"
PORTFOLIO_DIR = EVAL_DIR / "portfolio"

#: Figures are the exception: they are DELIVERABLES, not intermediates. They go
#: into the paper, they are small (a few MB), and they are worth reviewing in a
#: diff, so they stay in the repo and stay tracked regardless of
#: FACTORBENCH_OUT. Everything else above follows the env var to scratch.
FIGURE_DIR = Path("out/eval/figures")

#: Mining outputs are inputs, not artefacts, and stay in the repo: they are
#: small, they are version-controlled (factors.json), and they are what the
#: methods actually produced.
MINING_DIR = Path("out/mining")


def describe() -> str:
    """One line naming the active root, for a notebook or log to print.

    Also says whether the IC table is actually there. Reading an empty or stale
    tree silently is the single most expensive mistake in this pipeline, so the
    check is printed rather than left to the reader.
    """
    state = "ok" if IC_TABLE.exists() else "MISSING ic_table.csv"
    return f"eval artefacts: {EVAL_DIR}  (root from {_ROOT_SOURCE}) [{state}]"
