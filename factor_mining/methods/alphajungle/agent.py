"""The LLM-facing half of Alpha Jungle: portrait -> formula -> validation.

Implements Eq. 4-5 and Algorithm 1 lines 12-18. Generation is two-step
(Appendix D): the LLM first writes an *alpha portrait* — name, plain-language
rationale, and pseudo-code — and only then translates that portrait into a
concrete operation list with symbolic parameters plus up to three candidate
argument sets. Each argument set is backtested and the best is kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factor_mining.methods.alphajungle import grammar, prompts
from factor_mining.methods.alphajungle.evaluation import (
    AlphaMetrics,
    compute_metrics,
    dimension_scores,
)
from factor_mining.methods.alphajungle.fsa import Gene, describe
from factor_mining.methods.alphajungle.llm import (
    TEMPERATURE_CORRECT,
    TEMPERATURE_GENERATE,
    TEMPERATURE_SCORE,
    ChatClient,
    LLMError,
    complete_json,
)

# Appendix K, Figure 16 requirement 5.
MAX_ARGUMENT_SETS = 3
# Algorithm 1 lines 15-18 loop until valid; bounded so one bad node cannot
# consume the whole search budget (the paper states no bound — see METHOD.md).
MAX_CORRECTION_ROUNDS = 3


class GenerationError(RuntimeError):
    """The LLM could not produce a valid alpha within the correction budget."""


@dataclass
class Candidate:
    """A generated alpha: its portrait, its compiled tree, and its backtest."""

    expr: grammar.Expr
    name: str
    description: str
    metrics: AlphaMetrics
    signal: pd.Series
    arguments: dict[str, int] = field(default_factory=dict)

    @property
    def formula(self) -> str:
        return str(self.expr)


@dataclass
class Backtester:
    """Evaluates a candidate on the training window only."""

    panel: pd.DataFrame
    target: pd.Series
    train_mask: np.ndarray

    def run(self, expr: grammar.Expr, zoo_signals: list[pd.Series]) -> tuple[AlphaMetrics,
                                                                            pd.Series]:
        signal = grammar.signal(expr, self.panel)[self.train_mask]
        return compute_metrics(signal, self.target[self.train_mask], zoo_signals), signal


def _fields_block() -> str:
    return ", ".join(grammar.FIELDS)


def _portrait_block(portrait: dict) -> str:
    if not isinstance(portrait, dict):
        raise LLMError(f"expected a portrait object, got {type(portrait).__name__}")
    lines = [
        f"Name: {portrait.get('name', '')}",
        f"Description: {portrait.get('description', '')}",
        "Pseudo-code:",
    ]
    lines += [f"  {line}" for line in portrait.get("pseudo_code", [])]
    return "\n".join(lines)


class AlphaAgent:
    """Wraps the LLM calls that produce and repair alpha formulas."""

    def __init__(
        self,
        client: ChatClient,
        backtester: Backtester,
        window_range: tuple[int, int] = grammar.DEFAULT_WINDOW_RANGE,
    ) -> None:
        self.client = client
        self.backtester = backtester
        self.window_range = window_range

    # --- step 1: portraits -------------------------------------------------

    def seed_portrait(self, forbidden: list[Gene]) -> dict:
        """Figure 15 — a fresh alpha portrait to root a new search tree."""
        prompt = prompts.PORTRAIT_PROMPT.format(
            available_fields=_fields_block(),
            available_operators=grammar.operator_catalog(),
            freq_subtrees=describe(forbidden),
        )
        return complete_json(self.client, prompt, TEMPERATURE_GENERATE)

    def refinement_suggestion(
        self,
        dimension: str,
        candidate: Candidate,
        scores: dict[str, float],
        history: str,
        examples: list[Candidate],
        forbidden: list[Gene],
    ) -> dict:
        """Eq. 4 — a textual, dimension-targeted refinement suggestion."""
        if examples:
            block = "Effective alphas from the repository, for reference:\n" + "\n".join(
                f"- {e.name}: {e.formula}\n  rationale: {e.description}" for e in examples
            )
        else:
            # Appendix D: Turnover and Overfitting Risk are zero-shot, and the
            # repository is empty early on.
            block = "No reference alphas are provided for this dimension."
        score_table = "\n".join(f"- {k}: {v:.3f}" for k, v in scores.items())
        prompt = prompts.SUGGESTION_PROMPT.format(
            dimension=dimension,
            dimension_description=prompts.DIMENSION_DESCRIPTIONS[dimension],
            alpha_formula=candidate.formula,
            alpha_description=candidate.description,
            score_table=score_table,
            refinement_history=history or "(this is the root alpha)",
            examples_block=block,
            freq_subtrees=describe(forbidden),
        )
        return complete_json(self.client, prompt, TEMPERATURE_GENERATE)

    def refined_portrait(
        self, candidate: Candidate, suggestion: str, forbidden: list[Gene]
    ) -> dict:
        """Figure 18 — apply the suggestion, producing a new portrait."""
        prompt = prompts.REFINEMENT_PROMPT.format(
            available_fields=_fields_block(),
            available_operators=grammar.operator_catalog(),
            freq_subtrees=describe(forbidden),
            origin_alpha_formula=candidate.formula,
            refinement_suggestions=suggestion,
        )
        return complete_json(self.client, prompt, TEMPERATURE_GENERATE)

    # --- step 2: portrait -> formula, with correction ----------------------

    def compile_portrait(
        self, portrait: dict, zoo_signals: list[pd.Series], zoo_metrics: list[AlphaMetrics]
    ) -> Candidate:
        """Figure 16 plus Algorithm 1 lines 15-18 (validate, correct, retry)."""
        lo, hi = self.window_range
        prompt = prompts.FORMULA_PROMPT.format(
            available_fields=_fields_block(),
            available_operators=grammar.operator_catalog(),
            alpha_portrait_prompt=_portrait_block(portrait),
            window_range=f"[{lo}, {hi}]",
        )
        feedback = ""
        for attempt in range(MAX_CORRECTION_ROUNDS):
            temperature = TEMPERATURE_GENERATE if attempt == 0 else TEMPERATURE_CORRECT
            ask = prompt if not feedback else (
                f"{prompt}\n\nYour previous attempt was rejected. "
                f"Reason: {feedback}\nProduce a corrected formula."
            )
            try:
                payload = complete_json(self.client, ask, temperature)
                return self._best_argument_set(payload, portrait, zoo_signals, zoo_metrics)
            except (grammar.InvalidExpressionError, LLMError) as e:
                feedback = str(e)
        raise GenerationError(
            f"no valid formula after {MAX_CORRECTION_ROUNDS} attempts; last reason: {feedback}"
        )

    def _best_argument_set(
        self,
        payload: dict,
        portrait: dict,
        zoo_signals: list[pd.Series],
        zoo_metrics: list[AlphaMetrics],
    ) -> Candidate:
        """Backtest each argument set, keep the best (Appendix D).

        The paper says only "select the configuration yielding the best
        performance". Ranking is by the mean of the four *numeric* dimension
        scores — the LLM-scored Overfitting dimension is left out because it
        does not depend on the parameter values and would cost three extra LLM
        calls per expansion. See METHOD.md.
        """
        if not isinstance(payload, dict):
            raise grammar.InvalidExpressionError("Expected a JSON object with 'formula'.")
        operations = payload.get("formula")
        if not isinstance(operations, list):
            raise grammar.InvalidExpressionError("'formula' must be a list of operations.")
        argument_sets = payload.get("arguments") or [{}]
        if not isinstance(argument_sets, list):
            raise grammar.InvalidExpressionError("'arguments' must be a list of objects.")

        best: Candidate | None = None
        best_score = -np.inf
        errors: list[str] = []
        for arguments in argument_sets[:MAX_ARGUMENT_SETS]:
            if not isinstance(arguments, dict):
                errors.append("each entry of 'arguments' must be a JSON object")
                continue
            try:
                expr = grammar.from_operations(operations, arguments)
                grammar.validate(expr, window_range=self.window_range)
            except grammar.InvalidExpressionError as e:
                errors.append(str(e))
                continue
            metrics, signal = self.backtester.run(expr, zoo_signals)
            numeric = dimension_scores(metrics, zoo_metrics)
            score = float(np.mean(list(numeric.values())))
            if score > best_score:
                best_score = score
                best = Candidate(
                    expr=expr,
                    name=str(portrait.get("name") or "alpha"),
                    description=str(portrait.get("description") or ""),
                    metrics=metrics,
                    signal=signal,
                    arguments={k: int(v) for k, v in arguments.items()
                               if isinstance(v, (int, float))},
                )
        if best is None:
            raise grammar.InvalidExpressionError("; ".join(dict.fromkeys(errors))
                                                 or "no usable argument set")
        return best

    # --- step 3: overfitting risk -----------------------------------------

    def overfitting_score(self, candidate: Candidate, history: str) -> tuple[float, str]:
        """Figure 17 — the LLM's qualitative 0-10 generalization judgement."""
        prompt = prompts.OVERFITTING_PROMPT.format(
            alpha_formula=candidate.formula,
            refinement_history=history or "(no refinements; this is a seed alpha)",
        )
        try:
            payload = complete_json(self.client, prompt, TEMPERATURE_SCORE)
            score = float(payload["score"])
            reason = str(payload.get("reason", ""))
        except (LLMError, KeyError, TypeError, ValueError) as e:
            # A missing judgement should not abort the search; score neutrally
            # and record why (visible, not silent — see METHOD.md).
            return 5.0, f"(assessment unavailable: {e})"
        return float(np.clip(score, 0.0, 10.0)), reason
