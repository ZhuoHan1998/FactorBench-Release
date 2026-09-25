"""Run the official AlphaGen GP baseline (gplearn symbolic regression) on FactorBench data.

Mirrors ``gp.py`` from the AlphaGen repository (commit 259687e) — the genetic
programming baseline the AlphaGen paper compares against — with qlib
initialization replaced by the shared data contract and the hardcoded calendar
windows by ``DEFAULT_SPLIT``. All search hyperparameters are the official
values.

How the official script works, since it is non-obvious: gplearn is run over a
*one-row* design matrix whose single row holds the terminal **names**
(``close``, ``vwap``, ``Constant(1.0)``, ...). Every operator in
``alphagen_generic.operators`` is a gplearn function that concatenates its
inputs into an expression *string*, so a program's "output" is the text of an
AlphaGen expression. The fitness function then ``eval``s that text into an
``Expression`` and returns its training IC. GP therefore searches expression
text directly, and gplearn's crossover/mutation operate on the program tree.

The official target ``Ref($close, -20) / $close - 1`` is identical to the
shared contract's ``forward_return_20d``, so no target substitution is needed.
Emitted factors use the ``alphagen-v1`` grammar and are evaluated in-process by
the existing executor.

Usage:
    python -m factor_mining.methods.gp.run --market sp100 --seed 0
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import fire
import numpy as np
import torch

from factor_bench.data import DEFAULT_SPLIT, TARGET, load_panel
from factor_mining.contracts import ExpressionFactor, MiningRun, Provenance
from factor_mining.methods.alphagen.data import PanelStockData
from factor_mining.vendor import use_vendored

use_vendored("alphagen")

from alphagen.data.expression import (  # noqa: E402, F401
    EMA,
    WMA,
    Abs,
    Add,
    Constant,
    Corr,
    Cov,
    CSRank,
    Delta,
    Div,
    Feature,
    Greater,
    Kurt,
    Less,
    Log,
    Mad,
    Max,
    Mean,
    Med,
    Min,
    Mul,
    Operators,
    OutOfDataRangeError,
    Pow,
    Rank,
    Ref,
    Sign,
    Skew,
    Std,
    Sub,
    Sum,
    Var,
)
from alphagen.data.parser import (  # noqa: E402
    ExpressionParser,
    ExpressionParsingError,
)
from alphagen.models.linear_alpha_pool import MseAlphaPool  # noqa: E402
from alphagen.utils.random import reseed_everything  # noqa: E402
from alphagen_generic.operators import funcs as generic_funcs  # noqa: E402
from alphagen_qlib.calculator import QLibStockDataCalculator  # noqa: E402
from alphagen_qlib.stock_data import FeatureType  # noqa: E402
from gplearn.fitness import make_fitness  # noqa: E402
from gplearn.functions import make_function  # noqa: E402
from gplearn.genetic import SymbolicRegressor  # noqa: E402

OFFICIAL_REPO = "https://github.com/RL-MLDM/alphagen"
OFFICIAL_COMMIT = "259687e8f316994426416c530a94842a2fe6405e"

# Official gp.py: a program whose text has more than 20 parentheses is scored
# -1, which is how the script bounds expression length.
MAX_TOKEN_LENGTH = 20
# Official gp.py __main__ block.
CONSTANTS = (-30.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.01, 0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
FEATURE_TERMINALS = ("open_", "close", "high", "low", "volume", "vwap")


# A factor must have cross-sectional dispersion on at least this fraction of
# training days, or it cannot rank stocks and its IC is undefined.
MIN_LIVE_DAY_FRACTION = 0.95


def _live_day_fraction(value: "torch.Tensor") -> float:
    """Fraction of days whose cross-section has any dispersion at all.

    Counting distinct values does not work here: a legitimate binary factor
    (`Greater($close,$open)`) has two, while `Corr($low,0.01,30d)` is
    identically zero yet picks up ~10 distinct values of float noise across a
    whole window. Nor does testing the normalized signal — normalize_by_day
    divides by a ~0 std and amplifies that noise to O(1). Per-day spread,
    measured relative to the day's own scale, separates them cleanly.
    """
    finite = torch.isfinite(value)
    n = finite.sum(dim=1)
    high = torch.where(finite, value, torch.tensor(float("-inf"))).max(dim=1).values
    low = torch.where(finite, value, torch.tensor(float("inf"))).min(dim=1).values
    spread = high - low
    scale = torch.where(finite, value.abs(), torch.zeros_like(value)).sum(dim=1)
    scale = (scale / n.clamp(min=1)).clamp(min=1e-30)
    alive = (n >= 3) & torch.isfinite(spread) & (spread > 1e-9 * scale)
    return float(alive.float().mean())


def _contract_parser() -> ExpressionParser:
    """The exact parser ``factor_bench.eval.executors`` uses for alphagen-v1.

    Exporting an expression the shared executor cannot parse makes the factor
    unevaluable, so the export step round-trips every candidate through this.
    """
    return ExpressionParser(
        [*Operators, *[Sign, CSRank, Pow, Skew, Kurt, Rank]],
        ignore_case=True,
        additional_operator_mapping={"Max": [Greater], "Min": [Less], "Delta": [Sub]},
    )


def _terminal_namespace() -> dict[str, object]:
    """Names the fitness function's ``eval`` may resolve.

    The official script relies on ``from alphagen_generic.features import *``
    putting the feature singletons in module globals and calls a bare ``eval``.
    Here the namespace is built explicitly and restricted to the expression
    classes plus those singletons, so a program string can only ever denote an
    expression.
    """
    names: dict[str, object] = {
        "Abs": Abs, "Log": Log, "Add": Add, "Sub": Sub, "Mul": Mul, "Div": Div,
        "Greater": Greater, "Less": Less, "Ref": Ref, "Mean": Mean, "Sum": Sum,
        "Std": Std, "Var": Var, "Max": Max, "Min": Min, "Med": Med, "Mad": Mad,
        "Delta": Delta, "WMA": WMA, "EMA": EMA, "Cov": Cov, "Corr": Corr,
        "Constant": Constant,
    }
    # Official alphagen_generic.features aliases: open_ (open is a builtin).
    for feature in FeatureType:
        names[feature.name.lower() if feature.name != "OPEN" else "open_"] = Feature(feature)
    return names


def run_gp(
    market: str = "csi300",
    seed: int = 0,
    pool_capacity: int = 20,
    population_size: int = 1000,
    generations: int = 40,
    tournament_size: int = 600,
    mutual_ic_thres: float = 0.7,
    device: str = "cpu",
    out_dir: str = "out/mining",
    verbose: int = 1,
) -> Path:
    """Run the GP baseline on one market and write factors.json. Returns its path."""
    t0 = time.time()
    reseed_everything(seed)
    torch_device = torch.device(device)

    panel = load_panel(market)
    TARGET.validate(panel)  # leakage guard: cached target matches its definition

    # The search sees the training segment only.
    data_train = PanelStockData(panel, *DEFAULT_SPLIT.train, device=torch_device,
                                market=market)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1  # == the shared forward_return_20d
    calculator_train = QLibStockDataCalculator(data_train, target)

    save_path = Path(out_dir) / "gp" / f"{market}_{pool_capacity}_{seed}"
    save_path.mkdir(parents=True, exist_ok=True)

    namespace = _terminal_namespace()
    # Program text -> training IC. The official script uses this both as a
    # memo and as the harvest source: the pool is built from its best entries.
    cache: dict[str, float] = {}

    def _metric(y_true, y_pred, sample_weight) -> float:
        """Official ``gp.py::_metric``. ``y_pred`` is the program's text."""
        key = y_pred[0]
        if key in cache:
            return cache[key]
        token_len = key.count("(") + key.count(")")
        if token_len > MAX_TOKEN_LENGTH:
            return -1.0
        try:
            expr = eval(key, {"__builtins__": {}}, namespace)  # noqa: S307
            # Programs built only from constants (e.g. `Sum(2.0,40d)`) are
            # constant signals: IC is 0 and the alphagen-v1 parser refuses them
            # outright ("expects a featured expression"). Score them like an
            # over-long program and never cache them, so try_pool cannot pad
            # the pool with unevaluable factors. `is_featured` is upstream's
            # own check — its RL env requires it via is_valid().
            if not expr.is_featured:
                return -1.0
            ic = calculator_train.calc_single_IC_ret(expr)
        except OutOfDataRangeError:
            ic = -1.0
        except Exception:  # noqa: BLE001 - a malformed program is just unfit
            # The official script lets these propagate; on FactorBench panels a
            # program can produce a degenerate tensor, and killing a 40-
            # generation run over one bad individual is not useful. Counted
            # below so the rate stays visible.
            failures["n"] += 1
            return -1.0
        if np.isnan(ic):
            ic = -1.0
        cache[key] = ic
        return ic

    failures = {"n": 0}
    skipped: dict[str, int] = {"unparseable": 0, "degenerate": 0}
    metric = make_fitness(function=_metric, greater_is_better=True)
    funcs = [make_function(**func._asdict()) for func in generic_funcs]

    def build_pool(capacity: int, thres: float | None) -> MseAlphaPool:
        """Official ``gp.py::try_pool``: take the highest-IC cached programs,
        keeping only those whose mutual IC with the already-kept ones is within
        ``thres``, then load them into a linear pool."""
        pool = MseAlphaPool(
            capacity=capacity, calculator=calculator_train,
            ic_lower_bound=None, device=torch_device,
        )
        exprs: list[object] = []
        unparseable = {"n": 0}
        degenerate = {"n": 0}

        def acceptable(candidate) -> bool:
            if thres is None:
                return True
            return all(
                abs(pool.calculator.calc_mutual_IC(e, candidate)) <= thres for e in exprs
            )

        parser = _contract_parser()
        ranked = Counter(cache).most_common(capacity if thres is None else None)
        for key, _ic in ranked:
            try:
                candidate = eval(key, {"__builtins__": {}}, namespace)  # noqa: S307
            except Exception:  # noqa: BLE001
                continue
            if not candidate.is_featured:
                continue
            # gp.py's fitness evals program text directly, so it can score
            # expressions whose *sub*-operands are bare constants (e.g.
            # `Cov(2.0,Sum($high,20d),40d)`). The alphagen-v1 parser rejects
            # those ("expects a featured expression"), which would make the
            # exported factor unevaluable. Gate on the parser itself.
            try:
                parser.parse(str(candidate))
            except ExpressionParsingError:
                unparseable["n"] += 1
                continue
            # `Cov`/`Corr` against a constant operand parse fine but are
            # identically zero, so the expression contains a feature yet has no
            # cross-section to rank. Those are not factors: their IC is NaN
            # downstream and the pool assigns them weight 0 anyway. gp.py pads
            # the pool to capacity with them; we drop them instead (documented
            # deviation), measured on the training window only so nothing is
            # selected using valid/test data.
            try:
                value = candidate.evaluate(data_train)
            except (OutOfDataRangeError, RuntimeError, IndexError):
                degenerate["n"] += 1
                continue
            if _live_day_fraction(value) < MIN_LIVE_DAY_FRACTION:
                degenerate["n"] += 1
                continue
            if acceptable(candidate):
                exprs.append(candidate)
                if len(exprs) >= capacity:
                    break
        if exprs:
            pool.force_load_exprs(exprs)
        skipped["unparseable"] = unparseable["n"]
        skipped["degenerate"] = degenerate["n"]
        return pool

    def write_artifacts(complete: bool, generation: int) -> Path:
        """Persist the pool as it stands, every few generations rather than only
        at the end, so a long run that dies keeps what it mined."""
        pool = build_pool(pool_capacity, mutual_ic_thres)
        state = pool.state
        factors = [
            ExpressionFactor(
                name=f"gp_{market}_s{seed}_{i:02d}",
                expression=str(expr),
                grammar="alphagen-v1",
                weight=float(weight),
                train_metrics={"ic": float(ic)},
            )
            for i, (expr, weight, ic) in enumerate(
                zip(state["exprs"], state["weights"], state["ics_ret"])
            )
        ]
        mining_run = MiningRun(
            method="gp",
            paradigm="evolutionary",
            output_form="expression",
            market=market,
            seed=seed,
            split=DEFAULT_SPLIT.as_dict(),
            target=TARGET.name,
            search_budget={
                "population_size": population_size,
                "generations": generations,
                "generations_run": generation,
                "tournament_size": tournament_size,
                "pool_capacity": pool_capacity,
                "mutual_ic_thres": mutual_ic_thres,
                "max_token_length": MAX_TOKEN_LENGTH,
                "programs_evaluated": len(cache),
                "eval_failures": failures["n"],
                "skipped_unparseable": skipped["unparseable"],
                "skipped_degenerate": skipped["degenerate"],
                "min_live_day_fraction": MIN_LIVE_DAY_FRACTION,
                # bools coerce to 0.0 in this dict's value type
                "complete": str(complete).lower(),
            },
            wall_clock_seconds=time.time() - t0,
            provenance=Provenance(
                repo=OFFICIAL_REPO,
                commit=OFFICIAL_COMMIT,
                entry="gp.py",
                notes="Official AlphaGen GP baseline vendored unmodified; qlib "
                "data layer replaced by the FactorBench shared contract and the "
                "hardcoded calendar windows by DEFAULT_SPLIT. The official "
                "target Ref($close,-20)/$close - 1 is identical to the shared "
                "forward_return_20d. Pool harvested by gp.py's try_pool: "
                f"highest-IC programs with mutual IC <= {mutual_ic_thres}."
                + ("" if complete else " PARTIAL RUN: evolution stopped before "
                   "its generation budget; see search_budget.complete."),
            ),
            factors=factors,
        )
        written = mining_run.save(save_path / "factors.json")
        (save_path / "cache.json").write_text(json.dumps(
            {"programs": dict(Counter(cache).most_common(500)),
             "n_programs": len(cache), "eval_failures": failures["n"]},
            indent=2,
        ))
        return written

    generation_counter = {"n": 0}

    def on_generation(*_args, **_kwargs) -> None:
        """Official ``gp.py::ev`` — checkpoints every 4 generations."""
        generation_counter["n"] += 1
        if generation_counter["n"] % 4 != 0:
            return
        write_artifacts(complete=False, generation=generation_counter["n"])
        if verbose:
            best = max(cache.values()) if cache else float("nan")
            print(f"[gp] generation {generation_counter['n']}/{generations}: "
                  f"{len(cache)} programs cached, best train IC {best:.4f}, "
                  f"{failures['n']} eval failures")

    # Official __main__ block: one row of terminal names, a dummy target.
    terminals = list(FEATURE_TERMINALS) + [f"Constant({v})" for v in CONSTANTS]
    x_train = np.array([terminals])
    y_train = np.array([[1]])

    est_gp = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        init_depth=(2, 6),
        tournament_size=tournament_size,
        stopping_criteria=1.0,
        p_crossover=0.3,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.01,
        p_point_mutation=0.1,
        p_point_replace=0.6,
        max_samples=0.9,
        verbose=verbose,
        parsimony_coefficient=0.0,
        random_state=seed,
        function_set=funcs,
        metric=metric,  # type: ignore[arg-type]
        const_range=None,
        n_jobs=1,
    )
    est_gp.fit(x_train, y_train, callback=on_generation)

    path = write_artifacts(complete=True, generation=generation_counter["n"])
    if verbose:
        print(f"[gp] {len(cache)} programs evaluated "
              f"({failures['n']} failures) -> {path}")
    return path


if __name__ == "__main__":
    fire.Fire(run_gp)
