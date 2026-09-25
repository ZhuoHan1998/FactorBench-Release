#!/usr/bin/env python3
"""Collapse RD-Agent's per-round re-implementations to one factor per idea.

Why
---
RD-Agent's loop is Research -> Development -> Feedback: it proposes a factor,
implements it, reads feedback, and re-implements. ``loop.trace.hist`` is that
working history, and our harvester walks all of it, de-duplicating on the
*source string*. A rewrite changes the source without changing the numbers, so
each round survives as a separate factor. Four rounds of ``realized_vol_10d``
differing only in ``min_periods`` and index plumbing land in factors.json as
four factors with pairwise Spearman correlation of 1.000.

Measured on the harvest: 206 factors, 107 distinct factor names, and 77% of
factors have a within-run sibling at |Spearman| > 0.999 -- against 0-14% for
every other method. This is our extraction, not RD-Agent's behaviour, and it
inflates factor counts, RQ1's multiple-testing denominator, and RQ2's
within-method redundancy statistic.

AlphaAgent walks the same trace but keys its de-duplication on the *expression*
rather than the implementation, which is the correct key -- it needs no fix,
and neither does any other method.

The rule
--------
Names are "rdagent_<market>_s<seed>_r<round>_<factor name>". Group by the
factor name with the round index removed; keep the last round.

Last, because the alternatives are unavailable: ``accepted_by_feedback`` is 0.0
on all 206 factors (either the loop rejected everything or feedback.decision is
not read correctly -- we cannot tell without a diagnostic run), and selecting on
a train metric would be selecting on data. "What the agent finished with"
selects on nothing.

What it deliberately does NOT touch
-----------------------------------
Duplicates that are real mining behaviour: AlphaSage emitting one formula across
several seeds and markets, QuantaAlpha emitting two differently-named factors
with identical expressions. Those are findings RQ2 exists to measure.

Nothing in factor_mining/ changes -- the harvester and the vendored method stay
as they are, so what we ran remains what the official implementation does. This
is a separate, auditable pass over the artefacts.

Usage:
    python scripts/dedup_rdagent.py                 # dry run, writes nothing
    python scripts/dedup_rdagent.py --apply         # rewrite factors.json
    python scripts/dedup_rdagent.py --restore       # undo, from the backups
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

MINING_DIR = Path("out/mining/rdagent")
# Written the first time a run is rewritten and never overwritten afterwards,
# so the original harvest is always recoverable without going through git.
BACKUP = "factors.raw.json"
ROUND_RE = re.compile(r"^(?P<head>.+?_s\d+)_r(?P<round>\d+)_(?P<stem>.+)$")


def plan(factors: list[dict], keep: str = "last") -> tuple[list[dict], list[tuple[str, str]]]:
    """Split factors into (kept, [(dropped name, the name that supersedes it)]).

    A name without a round tag belongs to no group and is always kept.
    """
    groups: dict[str, list[tuple[int, int]]] = {}
    untagged: list[int] = []
    for i, f in enumerate(factors):
        m = ROUND_RE.match(f["name"])
        if m is None:
            untagged.append(i)
        else:
            groups.setdefault(m.group("stem"), []).append((int(m.group("round")), i))

    keep_ix = set(untagged)
    dropped: list[tuple[str, str]] = []
    for members in groups.values():
        members.sort()                                  # by round, then position
        chosen = members[-1] if keep == "last" else members[0]
        keep_ix.add(chosen[1])
        dropped += [(factors[i]["name"], factors[chosen[1]]["name"])
                    for _r, i in members if i != chosen[1]]

    return [f for i, f in enumerate(factors) if i in keep_ix], dropped


def source_path(run_dir: Path) -> Path:
    """Always plan from the original harvest, so the pass is idempotent."""
    backup = run_dir / BACKUP
    return backup if backup.exists() else run_dir / "factors.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mining-dir", default=str(MINING_DIR),
                    help=f"run directories to process (default: {MINING_DIR})")
    ap.add_argument("--keep", choices=["last", "first"], default="last")
    ap.add_argument("--apply", action="store_true",
                    help="write the deduplicated factors.json (default: dry run)")
    ap.add_argument("--restore", action="store_true",
                    help=f"put every {BACKUP} back as factors.json and stop")
    ap.add_argument("--verbose", action="store_true",
                    help="list every dropped factor and what supersedes it")
    args = ap.parse_args(argv)

    run_dirs = sorted(p.parent for p in Path(args.mining_dir).glob("*/factors.json"))
    if not run_dirs:
        print(f"no runs under {args.mining_dir}", file=sys.stderr)
        return 1

    if args.restore:
        n = 0
        for d in run_dirs:
            if (d / BACKUP).exists():
                shutil.copy2(d / BACKUP, d / "factors.json")
                n += 1
        print(f"restored {n} run(s) from {BACKUP}")
        return 0

    before = after = changed = 0
    for d in run_dirs:
        blob = json.loads(source_path(d).read_text())
        kept, dropped = plan(blob["factors"], args.keep)
        before += len(blob["factors"])
        after += len(kept)
        tag = "" if dropped else "   (nothing to drop)"
        print(f"{d.name:>16}: {len(blob['factors']):>3} -> {len(kept):>3}"
              f"   drop {len(dropped):>3}{tag}")
        if args.verbose:
            for gone, stays in dropped:
                print(f"{'':>18}- {gone}\n{'':>20}superseded by {stays}")

        if args.apply and dropped:
            if not (d / BACKUP).exists():       # never clobber the first harvest
                shutil.copy2(d / "factors.json", d / BACKUP)
            blob["factors"] = kept
            # The contract has no field for this, so the note goes in provenance,
            # which is free text and travels with the run.
            prov = blob.get("provenance")
            if isinstance(prov, dict):
                note = (f"scripts/dedup_rdagent.py: kept the {args.keep} round per "
                        f"factor name; dropped {len(dropped)} per-round "
                        f"re-implementations (original in {BACKUP})")
                prov["notes"] = f"{prov['notes']} | {note}" if prov.get("notes") else note
            (d / "factors.json").write_text(json.dumps(blob, indent=2))
            (d / "dedup_dropped.json").write_text(json.dumps(
                [{"dropped": g, "superseded_by": k} for g, k in dropped], indent=2))
            changed += 1

    pct = (1 - after / before) * 100 if before else 0.0
    print(f"\n{before} -> {after} factors ({before - after} dropped, {pct:.0f}%)")
    if args.apply:
        print(f"{changed} of {len(run_dirs)} runs rewritten; "
              f"ic_table / redundancy caches are now stale for rdagent")
    else:
        print("dry run -- nothing written. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
