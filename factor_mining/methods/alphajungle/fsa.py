"""Frequent Subtree Avoidance (paper Section 3, Eq. 11-12).

The repository's effective alphas are mined for *closed* frequent root genes;
the top-k most frequent become forbidden structural motifs that the generation
prompt tells the LLM to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

from factor_mining.methods.alphajungle.grammar import (
    Expr,
    render_abstract,
    root_genes,
    walk,
)

# Appendix G: "in the Frequent Subtree Avoidance method, the number of frequent
# subtrees to be avoided is set to 3."
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class Gene:
    """A forbidden structural motif and how common it is in the repository."""

    expr: Expr
    support: float

    @property
    def text(self) -> str:
        return render_abstract(self.expr)


def gene_support(zoo: list[Expr]) -> dict[Expr, float]:
    """Eq. 11: fraction of repository alphas whose gene set contains each gene."""
    if not zoo:
        return {}
    counts: dict[Expr, int] = {}
    for alpha in zoo:
        for gene in root_genes(alpha):
            counts[gene] = counts.get(gene, 0) + 1
    return {gene: n / len(zoo) for gene, n in counts.items()}


def _is_closed(gene: Expr, support: dict[Expr, float]) -> bool:
    """"A root gene is 'closed' if none of its immediate supertrees share the
    same support count" — i.e. it is a maximal pattern at its frequency."""
    own = support[gene]
    for other, other_support in support.items():
        if other is gene or other == gene:
            continue
        if other_support == own and gene in {n for n in walk(other)}:
            return False
    return True


def forbidden_structures(zoo: list[Expr], top_k: int = DEFAULT_TOP_K) -> list[Gene]:
    """Top-k most frequent *closed* root genes across the repository.

    Ties are broken by larger subtrees first, then by text, so the result is
    deterministic across runs with the same repository.
    """
    support = gene_support(zoo)
    if not support:
        return []
    closed = [g for g in support if _is_closed(g, support)]
    closed.sort(key=lambda g: (-support[g], -sum(1 for _ in walk(g)), render_abstract(g)))
    return [Gene(expr=g, support=support[g]) for g in closed[:top_k]]


def violates(expr: Expr, forbidden: list[Gene]) -> list[Gene]:
    """Eq. 12: which forbidden motifs (if any) the candidate contains."""
    genes = root_genes(expr)
    return [g for g in forbidden if g.expr in genes]


def describe(forbidden: list[Gene]) -> str:
    """The ``{freq_subtrees}`` prompt slot (paper Appendix K)."""
    if not forbidden:
        return "(none yet)"
    return "\n".join(f"- {g.text}" for g in forbidden)
