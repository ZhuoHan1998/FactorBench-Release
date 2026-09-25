"""The Multi-Agent Quality Checker (paper Section 3.3, Appendix A.3).

The paper's sequence is Code Quality -> Code Repair -> Judge -> Logic
Improvement -> execution and numerical-stability check -> temporal leakage unit
test. "All alpha codes that pass the quality checker are stored in the
candidate pool; otherwise, invalid codes are sent back to the multi-agent
system for repair. Codes that cannot be repaired or improved after several
attempts are discarded."

**Deviation — gate order.** We run execution/stability/leakage *before* the
Judge, not after. A candidate has to clear both to be admitted either way, so
the admitted set is unchanged; what changes is that a candidate which fails to
execute no longer costs a Judge call first. The Judge was 82% of all LLM calls
in the run at the default config, and against a slow endpoint that ordering was
the difference between a feasible run and an infeasible one. The one behavioural
consequence is that Logic Improvement now sees only executable code, and a
repair driven by an execution error is re-judged on the next attempt.

The deterministic parts of the Code Quality Agent's remit (parsing, undefined
names, forbidden imports, nested loops, static leakage) run in `codegen.py`
for free, before any LLM call. The LLM audit runs on top of that.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from factor_mining.llm import (
    TEMPERATURE_CORRECT,
    TEMPERATURE_SCORE,
    ChatClient,
    LLMError,
    complete_json,
)
from factor_mining.methods.cogalpha import prompts
from factor_mining.methods.cogalpha.codegen import (
    FactorCode,
    InvalidFactorCodeError,
    extract_functions,
    validate,
)
from factor_mining.methods.cogalpha.execution import (
    ExecutionError,
    ExecutionResult,
    Sandbox,
)

# Section 3.3: "Codes that cannot be repaired or improved after several
# attempts are discarded." The paper does not fix the number; 2 repair rounds
# keeps a bad candidate from consuming the run (see METHOD.md).
MAX_REPAIR_ROUNDS = 2


@dataclass
class CheckerStats:
    """Where candidates die, so a run can be diagnosed from one line."""

    seen: int = 0
    parse_failed: int = 0
    static_rejected: int = 0
    judge_rejected: int = 0
    execution_failed: int = 0
    leakage_rejected: int = 0
    repaired: int = 0
    passed: int = 0
    reasons: list[str] = field(default_factory=list)
    # Candidates are checked concurrently, and `+=` on an int attribute is not
    # atomic, so every update goes through `bump`.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False,
                                  compare=False)

    def bump(self, counter: str, reason: str | None = None) -> None:
        with self._lock:
            setattr(self, counter, getattr(self, counter) + 1)
            if reason is not None:
                self.reasons.append(reason)

    def as_dict(self) -> dict[str, int]:
        return {
            "seen": self.seen,
            "parse_failed": self.parse_failed,
            "static_rejected": self.static_rejected,
            "judge_rejected": self.judge_rejected,
            "execution_failed": self.execution_failed,
            "leakage_rejected": self.leakage_rejected,
            "repaired": self.repaired,
            "passed": self.passed,
        }


@dataclass
class CheckOutcome:
    """What the checker decided about one candidate, and why.

    The reason travels with the outcome rather than being read back off
    ``CheckerStats.reasons``: with candidates checked concurrently, the last
    entry in that shared list belongs to whichever thread appended most
    recently, not to this candidate.
    """

    factor: FactorCode | None = None
    result: ExecutionResult | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.factor is not None and self.result is not None


class QualityChecker:
    """Runs a raw LLM response through the checker into an executable factor."""

    def __init__(
        self,
        client: ChatClient,
        sandbox: Sandbox,
        judge: bool = True,
        llm_code_audit: bool = False,
        max_repair_rounds: int = MAX_REPAIR_ROUNDS,
    ) -> None:
        self.client = client
        self.sandbox = sandbox
        # Paper runs a semantic Judge on every candidate; it is the single
        # largest LLM cost in the pipeline, hence the switch.
        self.judge_enabled = judge
        # The deterministic checks cover most of the Code Quality Agent's
        # remit; the LLM audit on top is off by default for cost.
        self.llm_code_audit = llm_code_audit
        self.max_repair_rounds = max_repair_rounds
        self.stats = CheckerStats()

    # --- individual agents -------------------------------------------------

    def _code_quality(self, code: str) -> list[str]:
        payload = complete_json(
            self.client,
            prompts.CODE_QUALITY_PROMPT.format(code=code),
            TEMPERATURE_SCORE,
        )
        if isinstance(payload, dict) and not payload.get("passed", True):
            return [str(i) for i in payload.get("issues", [])][:5]
        return []

    def _repair(self, code: str, issues: str) -> str | None:
        raw = self.client.complete(
            prompts.CODE_REPAIR_PROMPT.format(
                code=code,
                issues=issues,
                requirements=prompts.REQUIREMENTS,
                libraries=prompts.LIBRARIES,
                output_format=prompts.OUTPUT_FORMAT,
            ),
            TEMPERATURE_CORRECT,
        )
        functions = extract_functions(raw)
        return functions[0] if functions else None

    def _judge(self, code: str) -> tuple[bool, list[str]]:
        payload = complete_json(
            self.client, prompts.JUDGE_PROMPT.format(code=code), TEMPERATURE_SCORE
        )
        if not isinstance(payload, dict):
            return True, []
        passed = bool(payload.get("passed", True))
        concerns = [str(c) for c in payload.get("concerns", [])][:5]
        if not passed and not concerns and payload.get("reason"):
            concerns = [str(payload["reason"])]
        return passed, concerns

    def _improve(self, code: str, concerns: list[str]) -> str | None:
        raw = self.client.complete(
            prompts.LOGIC_IMPROVEMENT_PROMPT.format(
                code=code,
                concerns="\n".join(f"- {c}" for c in concerns) or "- (unspecified)",
                complexity=prompts.COMPLEXITY_CONSTRAINTS,
                requirements=prompts.REQUIREMENTS,
                libraries=prompts.LIBRARIES,
                output_format=prompts.OUTPUT_FORMAT,
            ),
            TEMPERATURE_CORRECT,
        )
        functions = extract_functions(raw)
        return functions[0] if functions else None

    # --- the pipeline ------------------------------------------------------

    def check(self, source: str) -> CheckOutcome:
        """Push one candidate through every gate.

        Safe to call from several threads: the LLM client, the sandbox and the
        stats are each thread-safe, and no per-candidate state lives on ``self``.
        """
        self.stats.bump("seen")
        code = source
        for attempt in range(self.max_repair_rounds + 1):
            try:
                factor = validate(code)
            except InvalidFactorCodeError as e:
                problem = str(e)
                if attempt == self.max_repair_rounds:
                    self.stats.bump("static_rejected", f"static: {problem}")
                    return CheckOutcome(reason=f"static: {problem}")
                repaired = self._safe(lambda: self._repair(code, problem))
                if repaired is None:
                    reason = f"static (unrepaired): {problem}"
                    self.stats.bump("static_rejected", reason)
                    return CheckOutcome(reason=reason)
                code = repaired
                self.stats.bump("repaired")
                continue

            if self.llm_code_audit:
                issues = self._safe(lambda: self._code_quality(factor.source)) or []
                if issues and attempt < self.max_repair_rounds:
                    repaired = self._safe(
                        lambda: self._repair(factor.source, "\n".join(issues))
                    )
                    if repaired:
                        code = repaired
                        self.stats.bump("repaired")
                        continue

            try:
                result = self.sandbox.evaluate(factor)
            except ExecutionError as e:
                problem = str(e)
                leaked = "leakage" in problem
                if attempt == self.max_repair_rounds or leaked:
                    # A leaking factor is discarded outright: the paper accepts
                    # "only factors with zero leakage", and repairing a
                    # lookahead tends to produce a different factor anyway.
                    self.stats.bump(
                        "leakage_rejected" if leaked else "execution_failed",
                        f"execution: {problem}",
                    )
                    return CheckOutcome(reason=f"execution: {problem}")
                repaired = self._safe(lambda: self._repair(factor.source, problem))
                if repaired is None:
                    reason = f"execution (unrepaired): {problem}"
                    self.stats.bump("execution_failed", reason)
                    return CheckOutcome(reason=reason)
                code = repaired
                self.stats.bump("repaired")
                continue

            # The Judge is the most expensive gate in the pipeline and the only
            # one whose verdict does not depend on the numbers, so it runs last,
            # on candidates that already execute cleanly. See the note on
            # ordering at the top of this module.
            if self.judge_enabled:
                verdict = self._safe(lambda: self._judge(factor.source))
                passed, concerns = verdict if verdict else (True, [])
                if not passed:
                    if attempt == self.max_repair_rounds:
                        reason = "judge: " + (
                            concerns[0] if concerns else "semantic audit failed"
                        )
                        self.stats.bump("judge_rejected", reason)
                        return CheckOutcome(reason=reason)
                    improved = self._safe(lambda: self._improve(factor.source, concerns))
                    if improved:
                        code = improved
                        self.stats.bump("repaired")
                        continue

            self.stats.bump("passed")
            return CheckOutcome(factor=factor, result=result)

        return CheckOutcome(reason="no attempt produced an admissible factor")

    @staticmethod
    def _safe(call):
        """LLM hiccups should skip a gate, not abort the whole search."""
        try:
            return call()
        except LLMError:
            return None
