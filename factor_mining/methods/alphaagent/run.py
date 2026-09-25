"""Run the official AlphaAgent mining loop on the FactorBench contract.

Usage (from the repo root, inside the AlphaAgent venv):

    .venv-alphaagent/bin/python -m factor_mining.methods.alphaagent.run \
        --market csi300 --rounds 3 --direction "momentum and liquidity"

Mirrors ``alphaagent/app/qlib_rd_loop/factor_mining.py::main`` (commit 1da96e9)
minus its SIGALRM hard-kill wrapper, with the qlib backtest replaced by
FactorBench scoring via the official ``QLIB_FACTOR_RUNNER`` injection point.
The official Idea/Factor agents, FactorRegulator originality gate, DSL
executor, and feedback summarizer run unmodified.

LLM config comes from ``.env`` at the repo root. NOTE: this fork predates
RD-Agent's LiteLLM backend — it reads ``OPENAI_BASE_URL`` (not
OPENAI_API_BASE) and REQUIRES ``REASONING_MODEL`` (Idea/Factor/feedback
agents) in addition to ``CHAT_MODEL`` (expression repair + evaluators).
Embeddings are needed once factors start succeeding (CoSTEER RAG).
"""

from __future__ import annotations

import os
import platform
import threading
import time
from pathlib import Path

import fire

from factor_bench.data import DEFAULT_SPLIT, TARGET
from factor_mining.contracts import ExpressionFactor, MiningRun, Provenance
from factor_mining.methods.alphaagent.data_prep import prepare_data_folders

REPO_ROOT = Path(__file__).resolve().parents[3]

OFFICIAL_REPO = "https://github.com/RndmVariableQ/AlphaAgent"
OFFICIAL_COMMIT = "1da96e94a06a925c3997899f1848899440585efe"  # branch legacy-main

STEPS_PER_ROUND = 5  # factor_propose, factor_construct, factor_calculate, factor_backtest, feedback


def _setdefault_env(market: str, out_root: Path, full_dir: Path, debug_dir: Path) -> None:
    """Set AlphaAgent env vars; anything already in the environment/.env wins."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    # This fork reads OPENAI_BASE_URL; reuse an RD-Agent-style OPENAI_API_BASE
    # if that's what the user's .env provides.
    if "OPENAI_BASE_URL" not in os.environ and os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]
    # The "openai/" prefix is a LiteLLM routing convention (RD-Agent needs it);
    # this fork passes model names verbatim to the raw OpenAI SDK, so strip it.
    for var in ("CHAT_MODEL", "REASONING_MODEL", "EMBEDDING_MODEL"):
        value = os.environ.get(var, "")
        if value.startswith("openai/"):
            os.environ[var] = value.removeprefix("openai/")
    # REASONING_MODEL is required (no default in the official settings).
    if "REASONING_MODEL" not in os.environ and os.environ.get("CHAT_MODEL"):
        os.environ["REASONING_MODEL"] = os.environ["CHAT_MODEL"]

    defaults = {
        "USE_LOCAL": "True",
        "USE_AZURE": "False",
        # FactorBench component injection: official scen/proposal/summarizer
        # kept; the qlib backtest runner is replaced, and the coder is the
        # official FactorParser with a one-method fix for its crash when the
        # knowledge base has no successes yet (see coder_fix.py).
        "QLIB_FACTOR_RUNNER": "factor_mining.methods.alphaagent.components.FactorBenchFactorRunner",
        "QLIB_FACTOR_CODER": "factor_mining.methods.alphaagent.coder_fix.PatchedFactorParser",
        # Generated factor code reads these folders (linked into workspaces).
        "FACTOR_COSTEER_DATA_FOLDER": str(full_dir),
        "FACTOR_COSTEER_DATA_FOLDER_DEBUG": str(debug_dir),
        "FACTOR_COSTEER_PYTHON_BIN": str(REPO_ROOT / ".venv-alphaagent" / "bin" / "python"),
        # The runner cache key ignores the factor expression — unsafe.
        "CACHE_WITH_PICKLE": "False",
        "MULTI_PROC_N": "1",
        "WORKSPACE_PATH": str(out_root / "workspace"),
        "LOG_TRACE_PATH": str(out_root / "log"),
        "PICKLE_CACHE_FOLDER_PATH_STR": str(out_root / "pickle_cache"),
        "PROMPT_CACHE_PATH": str(out_root / "prompt_cache.db"),
        "FACTORBENCH_MARKET": market,
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _patch_darwin_symlink() -> None:
    """Official link_all_files_in_folder_to_workspace only links on Linux and
    Windows; on macOS nothing is linked and every factor fails. Patch the
    function at runtime (upstream RD-Agent later fixed this same bug)."""
    if platform.system() != "Darwin":
        return
    from alphaagent.core.experiment import FBWorkspace

    def _link_all(data_path, workspace_path) -> None:
        data_path, workspace_path = Path(data_path), Path(workspace_path)
        for f in data_path.iterdir():
            link = workspace_path / f.name
            if not link.exists():
                link.symlink_to(f)

    # Official helper is a staticmethod on FBWorkspace with Linux/Windows
    # branches only; rebind it on the class.
    FBWorkspace.link_all_files_in_folder_to_workspace = staticmethod(_link_all)


def harvest(loop, market: str, seed: int, rounds: int, wall: float, out_path: Path) -> Path:
    factors: list[ExpressionFactor] = []
    seen: set[str] = set()

    def add(task, round_idx: int, feedback=None, per_factor=None) -> None:
        expression = getattr(task, "factor_expression", None)
        if not expression or expression in seen:
            return
        seen.add(expression)
        metrics = dict((per_factor or {}).get(task.factor_name, {}))
        metrics["executed_ok"] = float(bool(getattr(task, "factor_implementation", False)))
        metrics["accepted_by_feedback"] = float(bool(feedback and feedback.decision))
        factors.append(
            ExpressionFactor(
                name=f"alphaagent_{market}_s{seed}_r{round_idx}_{task.factor_name}",
                expression=str(expression),
                grammar="alphaagent-v1",
                train_metrics=metrics,
            )
        )

    for round_idx, (hypothesis, exp, feedback) in enumerate(loop.trace.hist):
        per_factor = getattr(exp, "factorbench_metrics", {})
        for task in exp.sub_tasks:
            add(task, round_idx, feedback, per_factor)

    # A round that crashed or was skipped before its feedback step never
    # reaches trace.hist, but its constructed factors still live in the
    # loop's in-flight state — export them too (most-processed stage first).
    prev_out = getattr(loop, "loop_prev_out", {}) or {}
    for step in ("factor_backtest", "factor_calculate", "factor_construct"):
        exp = prev_out.get(step)
        if exp is not None and getattr(exp, "sub_tasks", None):
            per_factor = getattr(exp, "factorbench_metrics", {})
            for task in exp.sub_tasks:
                add(task, len(loop.trace.hist), None, per_factor)
            break
    run = MiningRun(
        method="alphaagent",
        paradigm="llm-agent",
        output_form="expression",
        market=market,
        seed=seed,
        split=DEFAULT_SPLIT.as_dict(),
        target=TARGET.name,
        search_budget={"rounds": rounds, "steps": rounds * STEPS_PER_ROUND},
        wall_clock_seconds=wall,
        provenance=Provenance(
            repo=OFFICIAL_REPO,
            commit=OFFICIAL_COMMIT,
            entry="alphaagent mine (AlphaAgentLoop), branch legacy-main",
            notes="Official Idea/Factor agents, FactorRegulator originality "
            "gate, DSL executor and summarizer unmodified; qlib backtest "
            "replaced by FactorBench valid-segment scoring on the shared "
            "contract. Official variable set ($open $high $low $close "
            "$volume $return) preserved.",
        ),
        factors=factors,
    )
    return run.save(out_path)


def run_alphaagent(
    market: str = "csi300",
    seed: int = 0,
    rounds: int = 3,
    direction: str | None = None,
    out_dir: str = "out/mining",
) -> Path:
    t0 = time.time()
    # Must be absolute: the official execute() runs `python <WORKSPACE_PATH>/
    # <uuid>/factor.py` with cwd set to that same workspace, so a relative
    # WORKSPACE_PATH gets resolved against itself and every execution fails.
    out_root = (Path(out_dir) / "alphaagent" / f"{market}_{rounds}_{seed}").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    full_dir, debug_dir = prepare_data_folders(market)
    TARGET  # target definition validated inside the runner via load_panel
    _setdefault_env(market, out_root, full_dir, debug_dir)

    # Import AFTER the environment is configured (import-time singletons).
    from alphaagent.app.qlib_rd_loop.conf import ALPHA_AGENT_FACTOR_PROP_SETTING
    from alphaagent.components.workflow.alphaagent_loop import AlphaAgentLoop

    _patch_darwin_symlink()

    loop = AlphaAgentLoop(
        ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction=direction,
        stop_event=threading.Event(),
        use_local=True,
    )
    try:
        loop.run(step_n=rounds * STEPS_PER_ROUND)
    finally:
        path = harvest(loop, market, seed, rounds, time.time() - t0, out_root / "factors.json")
        print(f"[alphaagent] factors -> {path}")
    return out_root / "factors.json"


if __name__ == "__main__":
    fire.Fire(run_alphaagent)
