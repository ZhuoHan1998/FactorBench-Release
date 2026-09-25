"""The seven-level agent hierarchy (paper Section 3.1, Appendix A.1, C.1).

Twenty-one task-specific agents, organised from macroscopic (Level I) to
microscopic (Level VII). Each agent differs from the shared base prompt in
exactly two slots — the expert persona and the "Factor Design Guidance"
block — which is how Appendix C.1 presents them ("Same as the Base Agent" for
every other section).

Personas and level descriptions are transcribed from Appendix A.1. Guidance
bodies are given verbatim in Appendix C.1 only for the agents the paper prints
(BarShape, Composite, and the base); for the rest the paper gives a one-line
exploration domain, and the guidance here is written from that line. Which is
which is recorded in ``Agent.guidance_source`` and summarised in METHOD.md.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

# Section 3.1: "The only raw factors available are open, high, low, close, and
# volume (OHLCV)." Note this excludes vwap, which the FactorBench panel has and
# other methods here use.
RAW_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

LEVELS: dict[int, str] = {
    1: "Market Structure & Cycle Layer",
    2: "Extreme Risk & Fragility Layer",
    3: "Price-Volume Dynamics Layer",
    4: "Price-Volatility Behavior Layer",
    5: "Multi-Scale Complexity Layer",
    6: "Stability & Regime-Gating Layer",
    7: "Geometric & Fusion Layer",
}


@dataclass(frozen=True)
class Agent:
    """One task-specific agent: a persona plus a design-guidance block."""

    name: str
    level: int
    persona: str
    guidance: str
    # "paper" = the guidance block is printed in Appendix C.1; "derived" = it is
    # written from the agent's one-line domain in Appendix A.1.
    guidance_source: Literal["paper", "derived"] = "derived"

    @property
    def layer(self) -> str:
        return LEVELS[self.level]


AGENTS: tuple[Agent, ...] = (
    # --- Level I: Market Structure & Cycle -------------------------------
    Agent(
        "AgentMarketCycle", 1,
        "an expert in **long-horizon market structure and cyclical analysis**",
        "Explore long-term cyclical transitions and phase shifts in price dynamics, "
        "revealing hidden market rhythms and structural turning points:\n"
        "- multi-horizon trend agreement and phase alignment;\n"
        "- distance from long-run moving averages, normalised by long-run range;\n"
        "- cycle-position estimates from rolling extrema or oscillator constructions;\n"
        "- persistence of a directional phase and the timing of its breakdown.",
    ),
    Agent(
        "AgentVolatilityRegime", 1,
        "an expert in **volatility regime detection and persistence**",
        "Detect transitions between calm and turbulent volatility states, "
        "characterizing regime persistence and clustering behavior:\n"
        "- ratios of short- to long-window realised volatility;\n"
        "- volatility-of-volatility and clustering measures;\n"
        "- time since the last regime switch, or a smooth regime indicator;\n"
        "- conditional behaviour of returns given the prevailing volatility state.",
    ),
    # --- Level II: Extreme Risk & Fragility -------------------------------
    Agent(
        "AgentTailRisk", 2,
        "an expert in **tail risk and downside sensitivity**",
        "Quantify downside sensitivity and tail-event exposure, modeling how "
        "negative shocks propagate through time:\n"
        "- rolling downside deviation and semi-variance;\n"
        "- empirical quantiles of the return distribution (e.g. 5th percentile);\n"
        "- magnitude and decay of the worst recent shocks;\n"
        "- asymmetry between the left and right tail of recent returns.",
    ),
    Agent(
        "AgentCrashPredictor", 2,
        "an expert in **crash precursors and structural fragility**",
        "Identify early warning signals of market collapses by tracking volatility "
        "compression, liquidity depletion, and structural fragility patterns:\n"
        "- volatility compression followed by range expansion;\n"
        "- volume drying up while price drifts;\n"
        "- widening gaps between intraday range and closing move;\n"
        "- accelerating negative skew or fragility build-up.",
    ),
    # --- Level III: Price-Volume Dynamics ---------------------------------
    Agent(
        "AgentLiquidity", 3,
        "an expert in **market liquidity and trading frictions**",
        "Measure market depth and trading frictions through price impact and "
        "turnover variability:\n"
        "- Amihud-style illiquidity: absolute return per unit of volume;\n"
        "- turnover variability and its trend;\n"
        "- price impact asymmetry between up and down days;\n"
        "- stability of the volume distribution over rolling windows.",
    ),
    Agent(
        "AgentOrderImbalance", 3,
        "an expert in **directional order-flow pressure inferred from OHLCV**",
        "Capture directional pressure from one-sided participation inferred from "
        "daily OHLCV patterns:\n"
        "- close position within the daily range as a buy/sell pressure proxy;\n"
        "- volume-weighted directional accumulation over rolling windows;\n"
        "- imbalance between up-volume and down-volume;\n"
        "- persistence and decay of one-sided pressure.",
    ),
    Agent(
        "AgentPriceVolumeCoherence", 3,
        "an expert in **price-volume synchronization and divergence**",
        "Examine synchronization and divergence between price and volume changes, "
        "revealing energy alignment or decoupling:\n"
        "- rolling correlation between returns and volume changes;\n"
        "- divergence between price trend and volume trend;\n"
        "- confirmation strength: moves supported vs. unsupported by volume;\n"
        "- decay of coherence across horizons.",
    ),
    Agent(
        "AgentVolumeStructure", 3,
        "an expert in **the statistical shape of trading activity**",
        "Analyze the statistical shape and concentration of trading activity to "
        "understand participation rhythm and clustering:\n"
        "- concentration of volume in a few days (entropy or Herfindahl-style);\n"
        "- skewness and kurtosis of the recent volume distribution;\n"
        "- ratio of volume spikes to baseline;\n"
        "- rhythm or periodicity in participation.",
    ),
    # --- Level IV: Price-Volatility Behavior ------------------------------
    Agent(
        "AgentDailyTrend", 4,
        "an expert in **directional persistence and multi-day momentum**",
        "Model directional persistence and multi-day momentum strength to uncover "
        "sustained price movements:\n"
        "- risk-adjusted momentum over several horizons;\n"
        "- fraction of recent days moving in the dominant direction;\n"
        "- trend efficiency: net move divided by total path length;\n"
        "- acceleration or deceleration of an established trend.",
    ),
    Agent(
        "AgentReversal", 4,
        "an expert in **short-term reversal and overreaction correction**",
        "Capture mean-reversion and short-term overreaction corrections following "
        "transient mispricings:\n"
        "- normalised distance from a short-run mean;\n"
        "- magnitude of recent extreme moves and their partial retracement;\n"
        "- reversal conditioned on the volume that accompanied the move;\n"
        "- asymmetry between reversal after gains and after losses.",
    ),
    Agent(
        "AgentRangeVol", 4,
        "an expert in **range-based volatility dynamics**",
        "Investigate range-based volatility dynamics, including compression-expansion "
        "cycles in daily price ranges:\n"
        "- Parkinson / Garman-Klass style range estimators from OHLC;\n"
        "- ratio of current range to its rolling norm;\n"
        "- compression-then-expansion sequences;\n"
        "- divergence between range volatility and close-to-close volatility.",
    ),
    Agent(
        "AgentLagResponse", 4,
        "an expert in **delayed price adjustment and lagged feedback**",
        "Study delayed price adjustments and lagged feedback between volatility, "
        "volume, and returns:\n"
        "- lead-lag correlation between volume shocks and subsequent returns;\n"
        "- autocorrelation structure of returns at several lags;\n"
        "- delayed response to a prior volatility shock;\n"
        "- speed of adjustment back to a reference level.",
    ),
    Agent(
        "AgentVolAsymmetry", 4,
        "an expert in **asymmetric volatility and skewed risk behaviour**",
        "Measure asymmetric volatility between upward and downward price moves, "
        "highlighting skewed risk behavior:\n"
        "- ratio of downside to upside realised volatility;\n"
        "- leverage-effect style coupling between returns and later volatility;\n"
        "- asymmetry of the intraday range on up vs. down days;\n"
        "- skewness of the recent return distribution.",
    ),
    # --- Level V: Multi-Scale Complexity -----------------------------------
    Agent(
        "AgentDrawdown", 5,
        "an expert in **drawdown geometry and recovery dynamics**",
        "Evaluate the depth, duration, and recovery geometry of cumulative losses, "
        "emphasizing temporal resilience:\n"
        "- current drawdown from a rolling peak, and its duration;\n"
        "- speed and completeness of recovery from prior drawdowns;\n"
        "- ratio of drawdown depth to the volatility that produced it;\n"
        "- underwater time as a fraction of the window.",
    ),
    Agent(
        "AgentFractal", 5,
        "an expert in **multi-scale roughness and long-memory structure**",
        "Assess multi-scale roughness and long-memory characteristics through "
        "cross-horizon variability and structural irregularity:\n"
        "- variance-ratio style comparisons across aggregation horizons;\n"
        "- path roughness: sum of absolute increments vs. net move;\n"
        "- Hurst-like scaling estimated from rolling ranges;\n"
        "- self-similarity or irregularity across nested windows.",
    ),
    # --- Level VI: Stability & Regime-Gating -------------------------------
    Agent(
        "AgentRegimeGating", 6,
        "an expert in **adaptive gating of signals by market state**",
        "Construct adaptive gates that modulate signal activation depending on "
        "volatility, trend, or liquidity states:\n"
        "- smooth gates (tanh, logistic) driven by a state variable;\n"
        "- a base signal scaled by a volatility or liquidity regime indicator;\n"
        "- conditional activation only in a favourable state;\n"
        "- blending two signals with a state-dependent weight.",
    ),
    Agent(
        "AgentStability", 6,
        "an expert in **temporal stability and signal robustness**",
        "Quantify temporal consistency and persistence in returns or derived "
        "signals, emphasizing robustness and smoothness:\n"
        "- rolling consistency of a signal's sign;\n"
        "- inverse of a signal's own volatility over the window;\n"
        "- smoothness measured as the roughness of successive differences;\n"
        "- agreement between the same construct at two window lengths.",
    ),
    # --- Level VII: Geometric & Fusion -------------------------------------
    Agent(
        "AgentBarShape", 7,
        "an expert in **candlestick geometry and bar-shape pattern analysis** "
        "using daily factors",
        "Translate candle geometry into quantitative signals:\n"
        "- ratios: (close-open)/(high-low), (high-close)/(close-low), etc.;\n"
        "- shadow asymmetry or balance indicators;\n"
        "- body-to-range normalization and persistence over recent days;\n"
        "- rolling geometry stability or asymmetry;\n"
        "- short-run shape momentum: recent trend in candle proportions.\n"
        "Encourage creativity and interpretability: derive smooth, bounded, "
        "differentiable functions using existing factors.",
        guidance_source="paper",
    ),
    Agent(
        "AgentCreative", 7,
        "an expert in **novel non-linear feature representations**",
        "Apply non-linear transformations, reparametrizations, or soft gating to "
        "generate novel feature representations:\n"
        "- bounded transforms (tanh, logistic, signed log) of raw constructs;\n"
        "- reparametrisations that change a signal's scale behaviour;\n"
        "- interactions between two unrelated constructs;\n"
        "- soft thresholding or saturation to tame outliers.",
    ),
    Agent(
        "AgentComposite", 7,
        "an expert in **composite factor construction and information fusion** "
        "using existing features",
        "Fuse signals through structured, interpretable transformations:\n"
        "- weighted or volatility-adjusted averages of trend, volume, and range "
        "features;\n"
        "- orthogonal combination: remove redundancy, amplify orthogonal content;\n"
        "- regime-weighted composites: dynamic weights based on volatility or "
        "liquidity states;\n"
        "- robust normalization before fusion (z-score or rank-scaling);\n"
        "- include non-linear combination terms (e.g., product, ratio) but keep "
        "compact.",
        guidance_source="paper",
    ),
    Agent(
        "AgentHerding", 7,
        "an expert in **crowding and directional consensus**",
        "Detect collective crowding behavior and directional alignment within "
        "OHLCV dynamics, reflecting market consensus intensity:\n"
        "- dispersion of returns relative to their own recent norm;\n"
        "- co-movement of price direction with volume surges;\n"
        "- persistence of one-directional consensus;\n"
        "- crowding intensity that builds then unwinds.",
    ),
)

AGENTS_BY_NAME: dict[str, Agent] = {a.name: a for a in AGENTS}


# Section 3.2 / Appendix A.2: five paraphrasing modes applied to an agent's
# guidance, to widen semantic coverage without leaving the analytical intent.
PARAPHRASE_MODES: dict[str, str] = {
    "light": (
        "Perform minimal rewording to maintain nearly identical meaning while "
        "improving clarity and linguistic fluency."
    ),
    "moderate": (
        "Rephrase the content naturally with mild enrichment or stylistic "
        "variation, capturing nuanced semantic differences under slightly "
        "altered descriptive framing."
    ),
    "creative": (
        "Introduce expressive, research-oriented rewording that adds "
        "interpretative depth, inspiring novel analytical angles or alternative "
        "reasoning patterns that remain aligned with the original domain."
    ),
    "divergent": (
        "Produce an exploratory rewrite from a new but relevant analytical "
        "viewpoint, shifting emphasis toward a different sub-mechanism within "
        "the same conceptual framework."
    ),
    "concrete": (
        "Make the guidance more specific and implementation-oriented by "
        "introducing measurable quantities such as statistical formulas, "
        "ratios, or example computations."
    ),
}


def select_initial_agents(rng: random.Random, n: int = 13) -> list[Agent]:
    """Appendix B.8: "adopt the golden ratio ... to randomly select 13 agents
    for constructing the initial factor pool" (21 x 0.618 ~= 13)."""
    return rng.sample(list(AGENTS), min(n, len(AGENTS)))


def agents_by_level() -> dict[int, list[Agent]]:
    out: dict[int, list[Agent]] = {level: [] for level in LEVELS}
    for agent in AGENTS:
        out[agent.level].append(agent)
    return out
