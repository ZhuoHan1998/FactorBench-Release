"""The factors.json contract: the sole interface from mining into evaluation.

Every mining method — regardless of search paradigm or output form — ends a
run by writing one ``factors.json`` conforming to :class:`MiningRun`. A factor
is either an *expression* in a named grammar (tree-based methods) or
*executable Python code* (code-based methods). Evaluation consumes only this
file; it never imports a method's internals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class ExpressionFactor(BaseModel):
    """A factor as an expression string in a named grammar (e.g. an AST)."""

    type: Literal["expression"] = "expression"
    name: str
    expression: str
    # Grammar identifier telling the evaluator which parser/compiler to use,
    # e.g. "alphagen-v1". New grammars must register an executor in
    # factor_bench before factors in them can be evaluated.
    grammar: str
    # Optional linear-combination weight when the method emits a weighted pool.
    weight: float | None = None
    train_metrics: dict[str, float] = Field(default_factory=dict)


class CodeFactor(BaseModel):
    """A factor as executable Python code over the canonical panel.

    Two entry-point conventions are supported:
    - ``entry_point="__main__"`` (RD-Agent's official contract): ``source`` is
      a script that reads ``daily_pv.h5`` (key ``"data"``, MultiIndex
      ``(datetime, instrument)``, canonical input fields) from its working
      directory and writes the signal to ``result.h5`` (key ``"data"``).
    - ``entry_point=<function name>``: ``source`` defines a function taking
      the canonical panel DataFrame and returning a pd.Series on its index.
    """

    type: Literal["code"] = "code"
    name: str
    language: Literal["python"] = "python"
    source: str
    entry_point: str = "factor"
    weight: float | None = None
    train_metrics: dict[str, float] = Field(default_factory=dict)


Factor = Annotated[Union[ExpressionFactor, CodeFactor], Field(discriminator="type")]


class Provenance(BaseModel):
    """Where the method implementation came from — official repo + pin."""

    repo: str
    commit: str
    entry: str = ""
    notes: str = ""


class MiningRun(BaseModel):
    """One mining run's output: metadata + the mined factors."""

    schema_version: Literal["1"] = "1"
    method: str
    paradigm: str
    output_form: Literal["expression", "code", "mixed"]
    market: str
    seed: int
    # Inclusive date bounds per segment, from factor_bench.data.splits.
    split: dict[str, tuple[str, str]]
    target: str
    search_budget: dict[str, float | int | str] = Field(default_factory=dict)
    wall_clock_seconds: float | None = None
    provenance: Provenance
    factors: list[Factor]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MiningRun":
        return cls.model_validate(json.loads(Path(path).read_text()))
