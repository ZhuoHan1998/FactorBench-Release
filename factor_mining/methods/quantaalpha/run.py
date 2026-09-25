"""Run the official QuantaAlpha evolutionary mining loop on FactorBench data.

Mirrors the official ``quantaalpha mine`` entry (commit b7ceb27) minus the CLI
wrapper's ``.env`` hard-exit: environment is configured here, an experiment
config with the official schema is written per run, and
``quantaalpha.pipeline.factor_mining.main`` runs unmodified. The qlib backtest
runner is replaced with FactorBench scoring via the official
``QLIB_FACTOR_RUNNER`` injection point.

Budget defaults are the paper's hyperparameters
(docs/experiment_hyperparameters.md: num_directions=10, max_rounds=11,
crossover_size=2, crossover_n=10) — a very large LLM budget; override for
smoke tests.

Usage (from the repo root, inside the QuantaAlpha venv):
    .venv-quantaalpha/bin/python -m factor_mining.methods.quantaalpha.run \
        --market csi300 --seed 0 --num_directions 2 --max_rounds 1
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import fire

from factor_bench.data import DEFAULT_SPLIT, TARGET
from factor_mining.contracts import ExpressionFactor, MiningRun, Provenance
from factor_mining.methods.quantaalpha.data_prep import prepare_data_folders
from factor_mining.vendor import VENDOR_ROOT

REPO_ROOT = Path(__file__).resolve().parents[3]

OFFICIAL_REPO = "https://github.com/QuantaAlpha/QuantaAlpha"
OFFICIAL_COMMIT = "b7ceb27b1001261d7a95b209a963664ae1f8ab23"

# The library manager writes repo-root-relative to the vendored tree.
_VENDOR_FACTORLIB = VENDOR_ROOT / "quantaalpha" / "data" / "factorlib"

_CONFIG_TEMPLATE = """\
planning:
  enabled: true
  use_llm: true
  allow_fallback: true
  max_attempts: 3
  num_directions: {num_directions}
execution:
  steps_per_loop: 5
  use_local: true
evolution:
  enabled: true
  mutation_enabled: true
  crossover_enabled: true
  max_rounds: {max_rounds}
  crossover_size: {crossover_size}
  crossover_n: {crossover_n}
  parallel_enabled: false
  fresh_start: true
quality_gate:
  consistency_enabled: true
"""


def _setdefault_env(market: str, out_root: Path, full_dir: Path, debug_dir: Path,
                    suffix: str) -> None:
    """Set QuantaAlpha env vars; anything already in the environment/.env wins."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    # This fork uses the raw OpenAI SDK (pre-LiteLLM): base-URL variable name
    # differs and model names must not carry LiteLLM's "openai/" prefix.
    if "OPENAI_BASE_URL" not in os.environ and os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]
    for var in ("CHAT_MODEL", "REASONING_MODEL", "EMBEDDING_MODEL"):
        value = os.environ.get(var, "")
        if value.startswith("openai/"):
            os.environ[var] = value.removeprefix("openai/")
    # The reasoning path is the DEFAULT in this fork (hypotheses, expressions,
    # mutation, crossover, planning, feedback all use REASONING_MODEL).
    if "REASONING_MODEL" not in os.environ and os.environ.get("CHAT_MODEL"):
        os.environ["REASONING_MODEL"] = os.environ["CHAT_MODEL"]

    defaults = {
        "USE_LOCAL": "True",
        # FactorBench component injection: official planning/proposal/coder/
        # quality-gates/evolution/summarizer kept; only the qlib backtest
        # runner is replaced.
        "QLIB_FACTOR_RUNNER": (
            "factor_mining.methods.quantaalpha.components.FactorBenchFactorRunner"
        ),
        # Conda-independent runtime probe (rdagent 0.8.0's crashes without
        # conda); belt-and-braces CONDA_DEFAULT_ENV for any residual callers.
        "QLIB_FACTOR_SCEN": (
            "factor_mining.methods.quantaalpha.components.FactorBenchScenario"
        ),
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV", "base"),
        # Factor code executes with this venv's interpreter.
        "FACTOR_CoSTEER_PYTHON_BIN": str(REPO_ROOT / ".venv-quantaalpha" / "bin" / "python"),
        # Generated factor.py reads these folders (linked into workspaces).
        "FACTOR_CoSTEER_DATA_FOLDER": str(full_dir),
        "FACTOR_CoSTEER_DATA_FOLDER_DEBUG": str(debug_dir),
        # Paper-value quality thresholds (docs/experiment_hyperparameters.md).
        "FACTOR_CoSTEER_SYMBOL_LENGTH_THRESHOLD": "250",
        # Caches are data/expression-blind — unsafe across benchmark runs.
        "CACHE_WITH_PICKLE": "False",
        "COSTEER_CODER_USE_CACHE": "False",
        # Keep every artifact under the run directory (absolute paths — the
        # official tree resolves relative ones inconsistently).
        "WORKSPACE_PATH": str(out_root / "workspace"),
        "PICKLE_CACHE_FOLDER_PATH_STR": str(out_root / "pickle_cache"),
        "FACTOR_CACHE_DIR": str(out_root / "factor_cache"),
        "LOG_TRACE_PATH": str(out_root / "log"),
        "PROMPT_CACHE_PATH": str(out_root / "prompt_cache.db"),
        "FACTOR_LIBRARY_SUFFIX": suffix,
        "FACTORBENCH_MARKET": market,
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _patch_darwin_symlink() -> None:
    """Inherited AlphaAgent bug: the official data-link helper only links on
    Linux/Windows, so factor execution fails on macOS. Patch at runtime;
    Linux (HPC) is unaffected."""
    import platform

    if platform.system() != "Darwin":
        return
    from quantaalpha.core.experiment import FBWorkspace

    def _link_all(data_path, workspace_path) -> None:
        data_path, workspace_path = Path(data_path), Path(workspace_path)
        for f in data_path.iterdir():
            link = workspace_path / f.name
            if not link.exists():
                link.symlink_to(f)

    FBWorkspace.link_all_files_in_folder_to_workspace = staticmethod(_link_all)


def _harvest(out_root: Path, market: str, seed: int, suffix: str, budget: dict,
             wall: float) -> Path:
    factors: list[ExpressionFactor] = []
    library_path = _VENDOR_FACTORLIB / f"all_factors_library_{suffix}.json"
    if library_path.exists():
        # Move the library out of the vendored tree into the run directory.
        target_path = out_root / library_path.name
        shutil.move(str(library_path), target_path)
        library = json.loads(target_path.read_text())
        for entry in (library.get("factors") or {}).values():
            expression = entry.get("factor_expression")
            if not expression:
                continue
            meta = entry.get("metadata") or {}
            feedback = entry.get("feedback") or {}
            metrics: dict[str, float] = {}
            if meta.get("round_number") is not None:
                metrics["round"] = float(meta["round_number"])
            metrics["accepted_by_feedback"] = float(bool(feedback.get("decision")))
            phase = meta.get("evolution_phase") or "na"
            factors.append(
                ExpressionFactor(
                    name=f"quantaalpha_{market}_s{seed}_{phase}_{entry.get('factor_name', 'f')}",
                    expression=str(expression),
                    grammar="quantaalpha-v1",
                    train_metrics=metrics,
                )
            )
    # Preserve the evolution lineage artifacts beside factors.json.
    for artifact in ("trajectory_pool.json", "evolution_state.json"):
        src = out_root / "log" / artifact
        if src.exists() and not (out_root / artifact).exists():
            shutil.copy(src, out_root / artifact)

    run = MiningRun(
        method="quantaalpha",
        paradigm="evolutionary-llm",
        output_form="expression",
        market=market,
        seed=seed,
        split=DEFAULT_SPLIT.as_dict(),
        target=TARGET.name,
        search_budget=budget,
        wall_clock_seconds=wall,
        provenance=Provenance(
            repo=OFFICIAL_REPO,
            commit=OFFICIAL_COMMIT,
            entry="quantaalpha mine (pipeline.factor_mining.main)",
            notes="Official planning/evolution/quality-gate pipeline unmodified; "
            "qlib backtest replaced by FactorBench valid-segment scoring on "
            "the shared contract. Official variable set preserved. Lineage in "
            "trajectory_pool.json / all_factors_library JSON beside this file.",
        ),
        factors=factors,
    )
    return run.save(out_root / "factors.json")


def run_quantaalpha(
    market: str = "csi300",
    seed: int = 0,
    direction: str = "daily price-volume alpha factors",
    num_directions: int = 10,
    max_rounds: int = 11,
    crossover_size: int = 2,
    crossover_n: int = 10,
    out_dir: str = "out/mining",
) -> Path:
    t0 = time.time()
    suffix = f"{market}_s{seed}"
    out_root = (Path(out_dir) / "quantaalpha" / f"{market}_{max_rounds}_{seed}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    full_dir, debug_dir = prepare_data_folders(market)
    _setdefault_env(market, out_root, full_dir, debug_dir, suffix)

    config_path = out_root / "experiment.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            num_directions=num_directions,
            max_rounds=max_rounds,
            crossover_size=crossover_size,
            crossover_n=crossover_n,
        )
    )
    budget = {
        "num_directions": num_directions,
        "max_rounds": max_rounds,
        "crossover_size": crossover_size,
        "crossover_n": crossover_n,
        "direction": direction,
    }

    # Import AFTER the environment is configured (import-time singletons).
    from quantaalpha.pipeline.factor_mining import main as quantaalpha_main

    _patch_darwin_symlink()
    try:
        quantaalpha_main(direction=direction, config_path=str(config_path))
    finally:
        path = _harvest(out_root, market, seed, suffix, budget, time.time() - t0)
        print(f"[quantaalpha] factors -> {path}")
    return out_root / "factors.json"


if __name__ == "__main__":
    fire.Fire(run_quantaalpha)
