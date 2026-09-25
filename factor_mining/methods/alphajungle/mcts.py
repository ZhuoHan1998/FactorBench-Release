"""LLM-guided MCTS over alpha formulas (paper Section 3, Algorithm 1).

One :class:`SearchTree` is Algorithm 1: seed an alpha, then repeatedly select a
node by UCT, expand it with an LLM refinement targeted at a weak evaluation
dimension, backtest the result, and backpropagate. :class:`AlphaJungleSearch`
runs trees back to back until the global node budget is spent, carrying the
effective-alpha repository and the FSA forbidden set across them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import pandas as pd

from factor_mining.methods.alphajungle import fsa
from factor_mining.methods.alphajungle.agent import (
    AlphaAgent,
    Candidate,
    GenerationError,
)
from factor_mining.methods.alphajungle.evaluation import (
    DIMENSIONS,
    AlphaMetrics,
    Evaluation,
    daily_cross_sectional_corr,
    dimension_probabilities,
    dimension_scores,
)
from factor_mining.methods.alphajungle.llm import LLMError

# Appendix G: exploration weight c = 1; initial search budget 3 per tree,
# incremented by 1 on every breakthrough; dimension-softmax temperature 1;
# 1 few-shot example; correlation filter eta = 50%.
EXPLORATION_WEIGHT = 1.0
INITIAL_BUDGET = 3
BUDGET_INCREMENT = 1
DIMENSION_TEMPERATURE = 1.0
N_FEW_SHOT = 1
CORRELATION_FILTER_PCT = 0.5


@dataclass
class Node:
    """One alpha in the search tree."""

    candidate: Candidate
    evaluation: Evaluation
    parent: "Node | None" = None
    children: list["Node"] = field(default_factory=list)
    visits: int = 0
    # Q(s, a) keyed by the child that action a leads to.
    q_values: dict[int, float] = field(default_factory=dict)
    dimension: str = ""
    suggestion: str = ""
    depth: int = 0

    @property
    def score(self) -> float:
        return self.evaluation.reward

    def lineage(self) -> list["Node"]:
        chain, node = [], self
        while node is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))


def refinement_history(node: Node) -> str:
    """Context for the LLM: this node's lineage plus its siblings and children.

    Section 3, Backpropagation: "the LLM is provided with rich contextual
    information: the refinement history of the current node's parent, children,
    and siblings. This allows the LLM to analyze the refinement trajectory and
    avoid redundant suggestions."
    """
    lines: list[str] = []
    for i, ancestor in enumerate(node.lineage()):
        prefix = "seed" if i == 0 else f"step {i} [{ancestor.dimension}]"
        lines.append(f"{prefix}: {ancestor.candidate.formula}")
        if ancestor.suggestion:
            lines.append(f"    suggestion applied: {ancestor.suggestion}")
        lines.append(
            "    scores: " + ", ".join(
                f"{d}={ancestor.evaluation.scores.get(d, 0.0):.2f}" for d in DIMENSIONS
            )
        )
    siblings = [s for s in (node.parent.children if node.parent else []) if s is not node]
    for sibling in siblings:
        lines.append(
            f"already tried from the same parent [{sibling.dimension}]: "
            f"{sibling.candidate.formula} (score {sibling.score:.3f})"
        )
    for child in node.children:
        lines.append(
            f"already tried from this alpha [{child.dimension}]: "
            f"{child.candidate.formula} (score {child.score:.3f})"
        )
    return "\n".join(lines) if lines else "(no refinements yet)"


class Repository:
    """The effective alpha repository F_zoo, plus the FSA forbidden set."""

    def __init__(self, top_k_genes: int = fsa.DEFAULT_TOP_K) -> None:
        self.candidates: list[Candidate] = []
        self.metrics: list[AlphaMetrics] = []
        self.forbidden: list[fsa.Gene] = []
        self._top_k_genes = top_k_genes

    def __len__(self) -> int:
        return len(self.candidates)

    @property
    def signals(self) -> list[pd.Series]:
        return [c.signal for c in self.candidates]

    def formulas(self) -> set[str]:
        return {c.formula for c in self.candidates}

    def add(self, candidate: Candidate) -> None:
        self.candidates.append(candidate)
        self.metrics.append(candidate.metrics)
        # Algorithm 1 line 30: refresh forbidden structures from the repository.
        self.forbidden = fsa.forbidden_structures(
            [c.expr for c in self.candidates], self._top_k_genes
        )

    def examples_for(self, dimension: str, candidate: Candidate,
                     k: int = N_FEW_SHOT) -> list[Candidate]:
        """Appendix D, "Refinement Suggestion Generation".

        Effectiveness/Stability: drop the top eta% most correlated alphas, then
        take the top-k by that dimension's metric. Diversity: take the k least
        correlated. Turnover/Overfitting Risk: zero-shot, no exemplars.
        """
        if dimension in ("turnover", "overfitting") or not self.candidates:
            return []
        scored = [
            (other, abs(daily_cross_sectional_corr(candidate.signal, other.signal)))
            for other in self.candidates
            if other.formula != candidate.formula
        ]
        if not scored:
            return []
        if dimension == "diversity":
            scored.sort(key=lambda pair: pair[1])
            return [c for c, _ in scored[:k]]
        scored.sort(key=lambda pair: pair[1])
        keep = max(1, int(len(scored) * (1.0 - CORRELATION_FILTER_PCT)))
        pool = [c for c, _ in scored[:keep]]
        metric = (lambda c: c.metrics.rank_ic) if dimension == "effectiveness" else (
            lambda c: c.metrics.rank_ir
        )
        pool.sort(key=metric, reverse=True)
        return pool[:k]


class SearchTree:
    """One MCTS tree — Algorithm 1, lines 2-36."""

    def __init__(
        self,
        root: Node,
        agent: AlphaAgent,
        repository: Repository,
        rng: random.Random,
        exploration_weight: float = EXPLORATION_WEIGHT,
        budget: int = INITIAL_BUDGET,
        budget_increment: int = BUDGET_INCREMENT,
    ) -> None:
        self.root = root
        self.agent = agent
        self.repository = repository
        self.rng = rng
        self.c = exploration_weight
        self.budget = budget
        self.budget_increment = budget_increment
        self.nodes: list[Node] = [root]
        self.best_score = root.score
        root.visits = 1

    # --- selection ---------------------------------------------------------

    def select(self) -> tuple[Node, list[Node]]:
        """Eq. 2, extended with the virtual expansion action a_e.

        "Unlike standard MCTS, which expands only leaf nodes, our approach
        allows any node to be selected for expansion" — the virtual action's
        visit count is 1 + |C(s)|. Its Q is the node's own alpha score, which
        the paper does not specify (see METHOD.md).
        """
        node = self.root
        path = [node]
        while True:
            if not node.children:
                return node, path
            log_n = math.log(max(node.visits, 1))
            best_child, best_value = None, -math.inf
            for index, child in enumerate(node.children):
                q = node.q_values.get(index, child.score)
                value = q + self.c * math.sqrt(log_n / max(child.visits, 1))
                if value > best_value:
                    best_value, best_child = value, child
            expand_value = node.score + self.c * math.sqrt(
                log_n / (1 + len(node.children))
            )
            if expand_value >= best_value or best_child is None:
                return node, path
            node = best_child
            path.append(node)

    # --- expansion ---------------------------------------------------------

    def choose_dimension(self, node: Node) -> str:
        """Eq. 3 — sample a target dimension, favouring the weakest."""
        probs = dimension_probabilities(node.evaluation.scores, DIMENSION_TEMPERATURE)
        names = list(probs)
        weights = [probs[n] for n in names]
        return self.rng.choices(names, weights=weights, k=1)[0]

    def expand(self, node: Node) -> Node | None:
        """Algorithm 1 lines 9-22. Returns the new node, or None on failure."""
        dimension = self.choose_dimension(node)
        history = refinement_history(node)
        examples = self.repository.examples_for(dimension, node.candidate)
        try:
            suggestion_payload = self.agent.refinement_suggestion(
                dimension, node.candidate, node.evaluation.scores, history,
                examples, self.repository.forbidden,
            )
            if not isinstance(suggestion_payload, dict):
                return None
            suggestion = str(suggestion_payload.get("suggestion", "")).strip()
            if not suggestion:
                return None
            portrait = self.agent.refined_portrait(
                node.candidate, suggestion, self.repository.forbidden
            )
            candidate = self.agent.compile_portrait(
                portrait, self.repository.signals, self.repository.metrics
            )
        except (LLMError, GenerationError):
            return None

        child_history = f"{history}\nproposed [{dimension}]: {suggestion}"
        overfitting, reason = self.agent.overfitting_score(candidate, child_history)
        scores = dimension_scores(candidate.metrics, self.repository.metrics, overfitting)
        child = Node(
            candidate=candidate,
            evaluation=Evaluation(candidate.metrics, scores, reason),
            parent=node,
            dimension=dimension,
            suggestion=suggestion,
            depth=node.depth + 1,
        )
        node.children.append(child)
        self.nodes.append(child)
        return child

    # --- backpropagation ---------------------------------------------------

    def backpropagate(self, path: list[Node], child: Node) -> None:
        """Eq. 9-10: visit counts up the path, Q as the subtree maximum."""
        chain = path + [child]
        reward = child.score
        for parent, nxt in zip(chain, chain[1:]):
            parent.visits += 1
            index = parent.children.index(nxt)
            parent.q_values[index] = max(parent.q_values.get(index, -math.inf), reward)
        child.visits += 1

    # --- driver ------------------------------------------------------------

    def run(self, max_nodes: int | None = None) -> list[Node]:
        """Expand until the dynamic budget is exhausted."""
        iteration = 0
        while iteration < self.budget:
            if max_nodes is not None and len(self.nodes) >= max_nodes:
                break
            iteration += 1
            node, path = self.select()
            child = self.expand(node)
            if child is None:
                continue
            self.backpropagate(path, child)
            # Algorithm 1 lines 32-35: a breakthrough buys more budget.
            if child.score > self.best_score:
                self.best_score = child.score
                self.budget += self.budget_increment
        return self.nodes
