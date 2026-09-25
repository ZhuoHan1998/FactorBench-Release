"""Run the official RD-Agent fin_factor loop on the FactorBench contract.

Usage (from the repo root, inside the RD-Agent venv):

    .venv-rdagent/bin/python -m factor_mining.methods.rdagent.loop \
        --market sp100 --loops 3

LLM access is configured via a ``.env`` file at the repo root (see
``factor_mining/methods/rdagent/env.example``). RD-Agent settings are
import-time singletons, so ALL environment variables are set here BEFORE the
first ``rdagent`` import — do not reorder the imports in ``run_rdagent``.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import fire

from factor_bench.data import DEFAULT_SPLIT, TARGET
from factor_mining.contracts import CodeFactor, MiningRun, Provenance
from factor_mining.methods.rdagent.data_prep import prepare_data_folders

REPO_ROOT = Path(__file__).resolve().parents[3]

OFFICIAL_REPO = "https://github.com/microsoft/RD-Agent"
OFFICIAL_COMMIT = "6762f84f9bc0f5c6486c50a00e128a57ac6c3683"


def _setdefault_env(market: str, out_root: Path, full_dir: Path, debug_dir: Path) -> None:
    """Set RD-Agent env vars; anything already in the environment/.env wins."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    defaults = {
        # LLM plumbing (fill real values in .env; see env.example)
        "BACKEND": "rdagent.oai.backend.LiteLLMAPIBackend",
        # FactorBench component injection: keep official proposal/coder,
        # replace the qlib-coupled runner + summarizer, and use a scenario
        # subclass whose runtime probe works without conda (HPC nodes).
        "QLIB_FACTOR_SCEN": "factor_mining.methods.rdagent.components.FactorBenchScenario",
        "QLIB_FACTOR_RUNNER": "factor_mining.methods.rdagent.components.FactorBenchFactorRunner",
        "QLIB_FACTOR_SUMMARIZER": (
            "factor_mining.methods.rdagent.components.FactorBenchExperiment2Feedback"
        ),
        # Shared chronological split (scenario prompt + experiment settings).
        "QLIB_FACTOR_TRAIN_START": DEFAULT_SPLIT.train[0],
        "QLIB_FACTOR_TRAIN_END": DEFAULT_SPLIT.train[1],
        "QLIB_FACTOR_VALID_START": DEFAULT_SPLIT.valid[0],
        "QLIB_FACTOR_VALID_END": DEFAULT_SPLIT.valid[1],
        "QLIB_FACTOR_TEST_START": DEFAULT_SPLIT.test[0],
        "QLIB_FACTOR_TEST_END": DEFAULT_SPLIT.test[1],
        # Generated factor.py reads these folders (symlinked into workspaces).
        "FACTOR_CoSTEER_DATA_FOLDER": str(full_dir),
        "FACTOR_CoSTEER_DATA_FOLDER_DEBUG": str(debug_dir),
        # Execute generated code with the research venv's interpreter.
        "FACTOR_CoSTEER_PYTHON_BIN": str(REPO_ROOT / ".venv" / "bin" / "python"),
        # The pickle cache keys on code only, not data — unsafe across markets.
        "CACHE_WITH_PICKLE": "False",
        # Scenario runtime-env probing expects a conda env name to exist.
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV", "base"),
        # Keep RD-Agent artifacts under the run's output directory.
        "WORKSPACE_PATH": str(out_root / "workspace"),
        "LOG_TRACE_PATH": str(out_root / "log"),
        # Which market the injected runner scores against.
        "FACTORBENCH_MARKET": market,
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def harvest(loop, market: str, seed: int, loops: int, wall_clock: float, out_path: Path) -> Path:
    """Extract every generated factor.py from the loop trace into factors.json."""
    factors: list[CodeFactor] = []
    seen: set[str] = set()
    for round_idx, (exp, feedback) in enumerate(loop.trace.hist):
        per_factor = getattr(exp, "factorbench_metrics", {})
        for task, ws in zip(exp.sub_tasks, exp.sub_workspace_list):
            if ws is None:
                continue
            source = ws.file_dict.get("factor.py")
            if not source or source in seen:
                continue
            seen.add(source)
            metrics = dict(per_factor.get(task.factor_name, {}))
            metrics["accepted_by_feedback"] = float(bool(feedback and feedback.decision))
            factors.append(
                CodeFactor(
                    name=f"rdagent_{market}_s{seed}_r{round_idx}_{task.factor_name}",
                    source=source,
                    entry_point="__main__",  # official contract: run file, read result.h5
                    train_metrics=metrics,
                )
            )
    run = MiningRun(
        method="rdagent",
        paradigm="llm-agent",
        output_form="code",
        market=market,
        seed=seed,
        split=DEFAULT_SPLIT.as_dict(),
        target=TARGET.name,
        search_budget={"loops": loops},
        wall_clock_seconds=wall_clock,
        provenance=Provenance(
            repo=OFFICIAL_REPO,
            commit=OFFICIAL_COMMIT,
            entry="rdagent fin_factor (FactorRDLoop)",
            notes="Official scenario/proposal/Co-STEER coder unmodified; qlib "
            "backtest runner and summarizer replaced by FactorBench IC scoring "
            "on the shared contract (train/valid only). Generated code follows "
            "the official contract: reads daily_pv.h5, writes result.h5.",
        ),
        factors=factors,
    )
    return run.save(out_path)


def run_rdagent(
    market: str = "sp100",
    seed: int = 0,
    loops: int = 3,
    out_dir: str = "out/mining",
) -> Path:
    t0 = time.time()
    out_root = Path(out_dir) / "rdagent" / f"{market}_{loops}_{seed}"
    out_root.mkdir(parents=True, exist_ok=True)
    full_dir, debug_dir = prepare_data_folders(market)
    _setdefault_env(market, out_root, full_dir, debug_dir)

    # Import AFTER the environment is fully configured (import-time singletons).
    from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop

    loop = FactorRDLoop(FACTOR_PROP_SETTING)
    try:
        asyncio.run(loop.run(loop_n=loops))
    finally:
        path = harvest(loop, market, seed, loops, time.time() - t0, out_root / "factors.json")
        print(f"[rdagent] factors -> {path}")
    return out_root / "factors.json"


if __name__ == "__main__":
    fire.Fire(run_rdagent)
