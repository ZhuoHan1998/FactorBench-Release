"""Thinking Evolution and the generation loop (paper Sections 3.2, 3.5, 3.6).

Each task-specific agent leads an evolutionary cycle. Within a generation the
parent pool is expanded by three operators — mutation only, crossover only, and
crossover followed by mutation (Section 3.6) — every child is pushed through
the Multi-Agent Quality Checker, scored, and the qualified survivors become the
next parent pool while the elite are banked.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factor_mining.llm import TEMPERATURE_GENERATE, ChatClient, LLMError
from factor_mining.methods.cogalpha import prompts
from factor_mining.methods.cogalpha.agents import (
    AGENTS,
    PARAPHRASE_MODES,
    Agent,
    select_initial_agents,
)
from factor_mining.methods.cogalpha.codegen import FactorCode, extract_functions, normalise
from factor_mining.methods.cogalpha.evaluation import Fitness, classify, compute_fitness
from factor_mining.methods.cogalpha.quality import QualityChecker

# Appendix B.4 defaults.
INITIAL_POOL_SIZE = 80
PARENT_POOL_SIZE = 32
CHILDREN_MULTIPLIER = 3
SUB_CYCLE_LENGTH = 8
INJECTION_EVERY = 2
CARRY_FORWARD_ELITE = 2
NUM_PER_REQUEST = 6
INITIAL_AGENT_COUNT = 13
# Requests in flight at once. The endpoint, not the algorithm, sets the pace:
# a run is thousands of independent LLM calls, each waiting on a slow remote
# model, so throughput scales with how many are outstanding. 1 restores the
# fully sequential behaviour.
CONCURRENCY = 8
PLATEAU_DELTA = 0.001
PLATEAU_WINDOW = 3

OPERATORS = ("mutation", "crossover", "crossover_mutation")


class BudgetExhaustedError(RuntimeError):
    """The configured LLM-call ceiling was reached; the run stops cleanly."""


@dataclass
class Candidate:
    """A surviving alpha: its code, its signal, and its fitness."""

    factor: FactorCode
    fitness: Fitness
    signal: pd.Series
    agent: str
    origin: str
    generation: int
    elite: bool = False

    @property
    def key(self) -> str:
        return normalise(self.factor.source)


@dataclass
class SearchConfig:
    """Paper defaults; ``generations`` is the one FactorBench shortens."""

    generations: int = 4
    parent_pool: int = PARENT_POOL_SIZE
    children_multiplier: int = CHILDREN_MULTIPLIER
    sub_cycle_length: int = SUB_CYCLE_LENGTH
    initial_pool: int = INITIAL_POOL_SIZE
    initial_agents: int = INITIAL_AGENT_COUNT
    num_per_request: int = NUM_PER_REQUEST
    injection_every: int = INJECTION_EVERY
    horizon: int = 20
    max_llm_calls: int = 0  # 0 = unlimited
    paraphrase: bool = True
    concurrency: int = CONCURRENCY


class CogAlphaSearch:
    """Drives the whole mining process for one market."""

    def __init__(
        self,
        client: ChatClient,
        checker: QualityChecker,
        target: pd.Series,
        train_mask: np.ndarray,
        market: str,
        rng: random.Random,
        config: SearchConfig,
        verbose: int = 1,
    ) -> None:
        self.client = client
        self.checker = checker
        self.target = target
        self.train_mask = train_mask
        self.market = market
        self.rng = rng
        self.cfg = config
        self.verbose = verbose

        self.elite: list[Candidate] = []
        self.seen: set[str] = set()
        self.calls = 0
        self._paraphrased: dict[str, str] = {}
        # Guard the two pieces of mutable state the worker threads touch.
        self._call_lock = threading.Lock()
        self._paraphrase_lock = threading.Lock()

    # --- LLM plumbing ------------------------------------------------------

    def _complete(self, prompt: str, temperature: float = TEMPERATURE_GENERATE) -> str:
        with self._call_lock:
            if self.cfg.max_llm_calls and self.calls >= self.cfg.max_llm_calls:
                raise BudgetExhaustedError(
                    f"reached the --max_llm_calls ceiling of {self.cfg.max_llm_calls}"
                )
            self.calls += 1
        return self.client.complete(prompt, temperature)

    def _map(self, fn, items: list):
        """Run ``fn`` over ``items`` concurrently, preserving input order.

        Every call is I/O-bound — a remote LLM request, or a sandbox
        subprocess — so threads give the throughput, and everything they touch
        (the client's usage counters, the checker's stats, the sandbox) is
        thread-safe. Anything drawn from ``self.rng`` is drawn by the caller
        before dispatch, so a seeded run reproduces at any ``concurrency``.
        """
        workers = min(max(1, self.cfg.concurrency), len(items))
        if workers <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))

    def paraphrase_mode(self) -> str:
        """Draw a paraphrase mode. Called on the main thread, before dispatch."""
        return self.rng.choice(list(PARAPHRASE_MODES))

    def _guidance(self, agent: Agent, mode: str | None = None) -> str:
        """Section 3.2: paraphrase the agent's guidance into one of five modes."""
        if not self.cfg.paraphrase:
            return agent.guidance
        mode = mode or self.paraphrase_mode()
        cache_key = f"{agent.name}:{mode}"
        with self._paraphrase_lock:
            if cache_key in self._paraphrased:
                return self._paraphrased[cache_key]
        if mode == "light":  # a light rewrite is not worth a call
            with self._paraphrase_lock:
                self._paraphrased[cache_key] = agent.guidance
            return agent.guidance
        try:
            text = self._complete(
                prompts.PARAPHRASE_PROMPT.format(
                    mode=mode,
                    mode_description=PARAPHRASE_MODES[mode],
                    guidance=agent.guidance,
                )
            ).strip()
        except (LLMError, BudgetExhaustedError):
            return agent.guidance
        text = text or agent.guidance
        with self._paraphrase_lock:
            self._paraphrased[cache_key] = text
        return text

    # --- Section 3.5: adaptive generation feedback -------------------------

    def _summarise(self, samples: list[tuple[str, str]]) -> str:
        if not samples:
            return prompts.NO_FEEDBACK
        block = "\n\n".join(
            f"Factor:\n{code}\nOutcome: {outcome}" for code, outcome in samples
        )
        try:
            return self._complete(prompts.SUMMARISE_PROMPT.format(samples=block)).strip()
        except (LLMError, BudgetExhaustedError):
            return prompts.NO_FEEDBACK

    def _feedback(
        self, valid: list[Candidate], invalid: list[tuple[str, str]]
    ) -> tuple[str, str]:
        """"we randomly select two valid alphas and two worst-performing
        invalid alphas as guiding samples"."""
        good = self.rng.sample(valid, min(2, len(valid))) if valid else []
        effective = self._summarise(
            [(c.factor.source, f"qualified, RankIC {c.fitness.rank_ic:+.4f}") for c in good]
        )
        ineffective = self._summarise(invalid[:2])
        return effective, ineffective

    # --- generation --------------------------------------------------------

    def _columns_desc(self, pool: list[Candidate]) -> tuple[int, str]:
        lines = ["date, ticker (MultiIndex)"] + [
            f"{f}: daily {f} price" if f != "volume" else "volume: daily traded volume"
            for f in ("open", "high", "low", "close", "volume")
        ]
        for candidate in pool[:20]:
            lines.append(f"{candidate.factor.name}: {candidate.factor.docstring[:110]}")
        return len(lines), "\n".join(f"- {line}" for line in lines)

    def _generate(self, agent: Agent, n: int, pool: list[Candidate],
                  effective: str, ineffective: str,
                  mode: str | None = None) -> list[str]:
        columns_num, columns_desc = self._columns_desc(pool)
        prompt = prompts.BASE_AGENT_PROMPT.format(
            persona=agent.persona,
            columns_num=columns_num,
            columns_desc=columns_desc,
            num_per_request=n,
            horizon=self.cfg.horizon,
            effective_cot=effective,
            ineffective_cot=ineffective,
            requirements=prompts.REQUIREMENTS,
            guidance=self._guidance(agent, mode),
            libraries=prompts.LIBRARIES,
            output_format=prompts.OUTPUT_FORMAT,
        )
        return extract_functions(self._complete(prompt))

    def _mutate(self, parent: Candidate, extra: str = "") -> list[str]:
        return extract_functions(self._complete(
            prompts.MUTATION_PROMPT.format(
                intro=f"The parent factor was produced by {parent.agent}.",
                complexity=prompts.COMPLEXITY_CONSTRAINTS,
                parent_factor_code=parent.factor.source,
                requirements=prompts.REQUIREMENTS,
                extra_guidance=extra,
                libraries=prompts.LIBRARIES,
                output_format=prompts.OUTPUT_FORMAT,
            )
        ))

    def _crossover(self, a: Candidate, b: Candidate, extra: str = "") -> list[str]:
        return extract_functions(self._complete(
            prompts.CROSSOVER_PROMPT.format(
                intro=f"The parents were produced by {a.agent} and {b.agent}.",
                complexity=prompts.COMPLEXITY_CONSTRAINTS,
                parent_factor_1_code=a.factor.source,
                parent_factor_2_code=b.factor.source,
                requirements=prompts.REQUIREMENTS,
                extra_guidance=extra,
                libraries=prompts.LIBRARIES,
                output_format=prompts.OUTPUT_FORMAT,
            )
        ))

    # --- scoring -----------------------------------------------------------

    def _admit(self, sources: list[str], agent: str, origin: str,
               generation: int) -> tuple[list[Candidate], list[tuple[str, str]]]:
        """Quality-check and score a batch. Returns (candidates, failures)."""
        admitted: list[Candidate] = []
        failures: list[tuple[str, str]] = []
        # The checker is the expensive part (a Judge call and a sandbox
        # subprocess per candidate) and is independent across candidates.
        # Scoring stays on this thread: it is pandas work, and keeping it here
        # makes `seen` and the admitted order independent of thread timing.
        outcomes = self._map(self.checker.check, list(sources))
        for source, outcome in zip(sources, outcomes, strict=True):
            if not outcome.ok:
                failures.append((source[:900], outcome.reason))
                continue
            key = normalise(outcome.factor.source)
            if key in self.seen:
                continue
            self.seen.add(key)
            signal = outcome.result.signal.reindex(self.target.index)
            fitness = compute_fitness(signal[self.train_mask], self.target[self.train_mask])
            admitted.append(Candidate(
                factor=outcome.factor, fitness=fitness, signal=signal,
                agent=agent, origin=origin, generation=generation,
            ))
        return admitted, failures

    def _select(self, population: list[Candidate]) -> tuple[list[Candidate],
                                                            list[Candidate]]:
        """Section 3.4: split a generation into qualified and elite."""
        pool = [c.fitness for c in population]
        qualified: list[Candidate] = []
        elite: list[Candidate] = []
        for candidate in population:
            is_qualified, is_elite, _ = classify(candidate.fitness, pool, self.market)
            if is_qualified:
                candidate.elite = is_elite
                qualified.append(candidate)
                if is_elite:
                    elite.append(candidate)
        qualified.sort(key=lambda c: c.fitness.score, reverse=True)
        return qualified[: self.cfg.parent_pool], elite

    # --- the loop ----------------------------------------------------------

    def build_initial_pool(self) -> list[Candidate]:
        """13 agents x ~6 alphas each, per Appendix B.8.

        The agents generate independently, as the paper describes them: each
        prompt describes the raw panel and no existing alphas. Doing it that way
        also makes the round exactly equivalent at any ``concurrency``, since
        no agent's prompt depends on what another agent produced.
        """
        agents = select_initial_agents(self.rng, self.cfg.initial_agents)
        per_agent = max(1, round(self.cfg.initial_pool / max(len(agents), 1)))
        # Drawn on this thread, before dispatch: `self.rng` is not thread-safe.
        tasks = [(agent, self.paraphrase_mode()) for agent in agents]

        def one(task: tuple[Agent, str]) -> tuple[list[str], Exception | None]:
            agent, mode = task
            try:
                return self._generate(
                    agent, per_agent, [], prompts.NO_FEEDBACK,
                    prompts.NO_FEEDBACK, mode=mode,
                ), None
            except (LLMError, BudgetExhaustedError) as e:
                return [], e

        population: list[Candidate] = []
        failures: list[tuple[str, str]] = []
        exhausted: BudgetExhaustedError | None = None
        for (agent, _), (sources, error) in zip(tasks, self._map(one, tasks),
                                                strict=True):
            if isinstance(error, BudgetExhaustedError):
                exhausted = error
            if not sources:
                continue
            admitted, failed = self._admit(sources, agent.name, "initial", 0)
            population.extend(admitted)
            failures.extend(failed)
        if self.verbose:
            print(f"[cogalpha] initial pool: {len(population)} alphas from "
                  f"{len(agents)} agents ({self.checker.stats.passed} passed checks)")
        self._last_failures = failures
        if exhausted is not None:
            # Everything the round already paid for is admitted first.
            raise exhausted
        return population

    def run(self) -> list[Candidate]:
        """Run every agent's evolutionary cycle. Returns the elite pool."""
        self._last_failures: list[tuple[str, str]] = []
        try:
            initial = self.build_initial_pool()
        except BudgetExhaustedError:
            return self.elite
        parents, elite = self._select(initial)
        self.elite.extend(elite)

        sub_cycles = max(1, -(-self.cfg.generations // self.cfg.sub_cycle_length))
        per_cycle = max(1, self.cfg.generations // sub_cycles)
        children_target = self.cfg.parent_pool * self.cfg.children_multiplier

        for agent in AGENTS:
            for cycle in range(sub_cycles):
                pool = list(parents)
                for step in range(per_cycle):
                    generation = cycle * per_cycle + step + 1
                    try:
                        pool = self._one_generation(
                            agent, pool, generation, children_target
                        )
                    except BudgetExhaustedError:
                        if self.verbose:
                            print("[cogalpha] LLM budget exhausted; stopping early")
                        return self.elite
                    if self._plateaued():
                        if self.verbose:
                            print(f"[cogalpha] {agent.name}: plateau, ending cycle")
                        break
                if pool:
                    parents = pool
        return self.elite

    def _one_generation(self, agent: Agent, parents: list[Candidate],
                        generation: int, children_target: int) -> list[Candidate]:
        if not parents:
            parents = self.build_initial_pool()
            if not parents:
                return []
        effective, ineffective = self._feedback(parents, self._last_failures)
        per_operator = max(1, children_target // len(OPERATORS))

        # Draw every parent up front, on this thread: `random.Random` is not
        # thread-safe, and drawing inside the workers would make the run depend
        # on thread scheduling instead of on the seed.
        tasks: list[tuple[str, tuple[Candidate, ...]]] = []
        for operator in OPERATORS:
            for _ in range(max(1, per_operator // self.cfg.num_per_request)):
                if operator == "mutation":
                    tasks.append((operator, (self.rng.choice(parents),)))
                else:
                    tasks.append((operator, tuple(self._two_parents(parents))))

        def produce(task: tuple[str, tuple[Candidate, ...]]) -> tuple[str, list[str]]:
            operator, chosen = task
            try:
                if operator == "mutation":
                    produced = self._mutate(chosen[0])
                else:
                    produced = self._crossover(*chosen)
            except LLMError:
                return operator, []
            if operator == "crossover_mutation" and produced:
                # Section 3.6: "crossover followed by mutation".
                outcome = self.checker.check(produced[0])
                if outcome.ok:
                    intermediate = Candidate(
                        factor=outcome.factor, fitness=compute_fitness(
                            outcome.result.signal.reindex(
                                self.target.index)[self.train_mask],
                            self.target[self.train_mask],
                        ),
                        signal=outcome.result.signal, agent=agent.name,
                        origin=operator, generation=generation,
                    )
                    try:
                        produced = self._mutate(intermediate)
                    except LLMError:
                        pass
            return operator, produced

        sources: list[tuple[str, str]] = []
        for operator, produced in self._map(produce, tasks):
            sources.extend((s, operator) for s in produced)

        # Section 3.5 / B.4: inject fresh agent alphas every few generations.
        if generation % self.cfg.injection_every == 0:
            try:
                fresh = self._generate(
                    agent, self.cfg.num_per_request, parents, effective, ineffective
                )
                sources.extend((s, "injection") for s in fresh)
            except LLMError:
                pass

        population: list[Candidate] = []
        failures: list[tuple[str, str]] = []
        for origin in OPERATORS + ("injection",):
            batch = [s for s, o in sources if o == origin]
            if not batch:
                continue
            admitted, failed = self._admit(batch, agent.name, origin, generation)
            population.extend(admitted)
            failures.extend(failed)
        self._last_failures = failures

        # "the top two elite alphas from the previous generation are always
        # carried forward to the next" (Section 3.4).
        population.extend(sorted(parents, key=lambda c: c.fitness.score,
                                 reverse=True)[:CARRY_FORWARD_ELITE])
        qualified, elite = self._select(population)
        self.elite.extend(e for e in elite if e.key not in {x.key for x in self.elite})
        if self.verbose:
            best = max((c.fitness.rank_ic for c in population), default=float("nan"))
            print(f"[cogalpha] {agent.name} gen {generation}: {len(population)} scored, "
                  f"{len(qualified)} qualified, {len(elite)} elite, "
                  f"elite pool {len(self.elite)}, best RankIC {best:+.4f}, "
                  f"{self.calls} LLM calls")
        return qualified

    def _two_parents(self, parents: list[Candidate]) -> tuple[Candidate, Candidate]:
        if len(parents) == 1:
            return parents[0], parents[0]
        a, b = self.rng.sample(parents, 2)
        return a, b

    def _plateaued(self) -> bool:
        """Appendix B.4 plateau rule on the elite pool's mean score."""
        window = PLATEAU_WINDOW
        scores = [c.fitness.score for c in self.elite]
        if len(scores) < 2 * window:
            return False
        prev = float(np.mean(scores[-2 * window: -window]))
        curr = float(np.mean(scores[-window:]))
        return (curr - prev) <= PLATEAU_DELTA


@dataclass
class SearchOutcome:
    elite: list[Candidate] = field(default_factory=list)
    llm_calls: int = 0
    checker_stats: dict[str, int] = field(default_factory=dict)
