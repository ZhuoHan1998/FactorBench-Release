"""Prompts, transcribed from the paper's Appendix C.

Appendix C.1 prints the Base Agent prompt in full and then gives each
task-specific agent as a delta ("Same as the Base Agent" for every section
except the persona and the Factor Design Guidance). That structure is
reproduced here: shared blocks are module constants, and `agents.py` supplies
the two per-agent slots.

C.2 gives the four Multi-Agent Quality Checker prompts, C.3 the Crossover and
Mutation prompts. Rendered from prose figures, so bullets become dashes.

Two substitutions are ours, both forced by this environment and recorded in
METHOD.md: the library list states the versions actually installed here, and
`talib` is removed because it is not available (it needs a system C library).
"""

from __future__ import annotations

# Appendix C.1, "Requirements".
REQUIREMENTS = """\
### Requirements:
- The input `DataFrame` has a MultiIndex of (date, ticker), and has already \
been grouped by ticker:
  - Each input `DataFrame` is a time series of a single stock.
- Output: A `pd.Series` indexed by `(date, ticker)` with the same name as the \
function.
- Each function must:
  - Have a descriptive, unique name: \
factor_<logic>_<transformation(s)>_<window(s)>_<field>.
  - Include a clear docstring explaining the logic and formula.
  - Balance predictive power with economic/financial interpretability.
  - Use an output column name that exactly matches the function name.
  - Be concise, precise, and readable.
  - Build new alpha factors based on existing ones."""

# Appendix C.1, "Pre-imported libraries" + "Coding Guidelines". Versions are
# this environment's; talib is dropped (unavailable here) — see METHOD.md.
LIBRARIES = """\
### Pre-imported libraries you can use (current versions):
- "np": import numpy as np (numpy version: 2.4)
- "pd": import pandas as pd (pandas version: 3.0)
- "stats": from scipy import stats (scipy version: 1.17)
- "math": import math (built-in module)
Do NOT import anything else. In particular talib is NOT available; implement \
technical constructs directly with pandas/numpy.

Coding Guidelines:
- Ensure the code is robust, efficient, and optimized:
  - Handle edge cases and exceptions (e.g., NaN values).
  - Minimize unnecessary computations and prefer vectorized operations.
  - Ensure numerical stability.
  - Strict Rule: Nested loops are absolutely forbidden.
    - You must never write any form of loop inside another loop.
    - Forbidden patterns include but are not limited to: for inside for, \
while inside while, for inside while, while inside for.
    - Any nested iteration structure is prohibited, regardless of indentation \
depth.
    - The use of while True or any potentially infinite loop is strictly \
prohibited.
- When filtering or assigning values in a DataFrame, always use \
df_copy.loc[row_indexer, col_indexer] = value.
- Code should be clean, maintainable, and efficient for large datasets:
  - Use descriptive variable names and minimize memory usage.
  - Avoid creating unnecessary copies of large DataFrames."""

# Appendix C.1, "Output format specification".
OUTPUT_FORMAT = """\
### Output format specification:
- Do NOT use markdown (like ```python).
- Do NOT add any explanation or comments outside the function.
- Each function must be wrapped inside: <<function N>> ... <</function N>>.
- All generated code must be executable and numerically stable.
- Always define intermediate columns (e.g., df_copy['x']) before referencing \
them later.
- The returned Series must be named exactly the same as the function name.
- Each function should follow this format:

<<function 1>>
def factor_xyz(df):
    \"\"\"Explain the logic. One clear idea. Short formula. No redundant stacking.\"\"\"
    df_copy = df.copy()
    # factor computation
    return df_copy["factor_xyz"]
<</function 1>>"""

# Appendix C.3, "Hard Complexity Constraints (must-follow)".
COMPLEXITY_CONSTRAINTS = """\
### Hard Complexity Constraints (must-follow)
Remember: Simple factors are often the most powerful and stable.
- Single theme, minimal path: each factor must represent one clear idea.
- Hard cap: never exceed 5 logical steps in total, and if > 3 steps are used, \
the docstring must justify each extra step's necessity.
- No redundancy / nesting: forbid stacked or decorative transforms (e.g., \
zscore(zscore(x)), rank(rank(x)), deep EMA chains without rationale).
- No theme mixing: do not combine unrelated ideas.
- Avoid nested or layered operations.
- Avoid unnecessary complexity or logic stacking."""


# --- Appendix C.1: the Base Agent generation prompt ------------------------

BASE_AGENT_PROMPT = """\
You are {persona}. Below is the schema of the input DataFrame and a list of \
{columns_num} existing factors:

{columns_desc}

The input DataFrame consists of daily aggregated factors — i.e., each row \
represents a single trading day's features for a given stock, already \
aggregated to daily frequency.
Please generate {num_per_request} new and original quantitative factor \
functions that are distinct from the existing ones, to forecast \
{horizon}-day forward returns. Each factor should be implemented as a \
complete Python function.

---
### Analysis of Effective Factors and Innovation Directions:
Below is a condensed CoT-style summary built from recent successful cases, \
explaining why they work well.
Mini-Chain from Survivors (Observation -> Cause -> Fix):
{effective_cot}

Based on these strengths, focus on incorporating similar principles in new \
factor creation.
Seek innovative methods to generate more efficient, robust, and adaptable \
factors, ensuring they work well in diverse market conditions while avoiding \
look-ahead/leakage and redundancy.

---
### Analysis of Ineffective Factors and Innovation Directions:
Below is a condensed CoT-style summary built from recent failure cases, \
explaining why they fail.
Mini-Chain from Failures (Observation -> Cause -> Fix):
{ineffective_cot}

Based on these failures, focus on avoiding similar issues in new factor \
creation.
Seek innovative methods to generate more effective, robust, and adaptable \
factors, ensuring they work well in diverse market conditions.

---
{requirements}

---
### Factor Design Guidance:
You are encouraged to explore a wide variety of signals and techniques, \
including but not limited to:
{guidance}

Please do NOT limit yourself to simple formulas or common patterns.
You are expected to innovate, introduce mathematically sophisticated or \
unconventional structures, and combine multiple concepts where reasonable.
The goal is to generate factors that are predictive, robust, and economically \
interpretable, while being structurally diverse from existing factors.

---
{libraries}

---
{output_format}"""


# --- Appendix A.2 / Section 3.2: guidance paraphrasing ---------------------

PARAPHRASE_PROMPT = """\
Rewrite the following alpha-factor design guidance in the "{mode}" style.
{mode_description}

Preserve the original analytical intent and the domain it covers. Do not add \
requirements, formatting rules, or output-format instructions.

Original guidance:
{guidance}

Respond with the rewritten guidance only, as a plain bulleted list."""


# --- Appendix C.2: Multi-Agent Quality Checker ----------------------------

CODE_QUALITY_PROMPT = """\
You are a **Code Quality Agent** auditing a generated alpha-factor function.
Perform a first-pass audit of the code below. Detect syntactic errors, \
undefined variables, formatting inconsistencies, invalid library calls, and \
potential runtime failures.

### Factor code:
{code}

### Checks:
- Does the function parse, and is every referenced name defined before use?
- Are only np, pd, stats and math used? (talib is NOT available.)
- Are intermediate columns created before being referenced?
- Does it return a Series named exactly like the function?
- Are there nested loops or unbounded while loops? (Both are forbidden.)
- Are there obvious runtime hazards (division by zero, log of a \
non-positive quantity, ambiguous truth values)?

Respond with JSON only:
{{"passed": true or false, "issues": ["one short sentence per issue"]}}"""

CODE_REPAIR_PROMPT = """\
You are a **Code Repair Agent**. Fix the alpha-factor function below so it is \
syntactically and operationally viable, without changing its core hypothesis.
Repairs include correcting import statements, rewriting malformed \
expressions, resolving type mismatches, and rewriting unstable numerical \
operations.

### Factor code:
{code}

### Issues reported:
{issues}

---
{requirements}

---
{libraries}

---
{output_format}"""

JUDGE_PROMPT = """\
You are a **Judge Agent** evaluating an alpha factor at the semantic level.
Assess whether the factor is:
- Logically consistent: correct operator ordering, coherent data flow, no \
degenerate expressions;
- Technically correct: valid use of rolling windows and transforms;
- Economically meaningful: obeys financial intuition and avoids fabricated \
indicators.

### Factor code:
{code}

Respond with JSON only:
{{"passed": true or false, "reason": "one sentence", \
"concerns": ["one short sentence per concern"]}}"""

LOGIC_IMPROVEMENT_PROMPT = """\
You are a **Logic Improvement Agent**. The factor below failed a semantic \
audit. Refine it: restructure formulas, adjust window parameters, replace \
dubious transformations, eliminate redundant operations, and enhance \
financial interpretability — while preserving the original modeling intent.

### Factor code:
{code}

### Judge's concerns:
{concerns}

---
{complexity}

---
{requirements}

---
{libraries}

---
{output_format}"""


# --- Appendix C.3: Thinking Evolution --------------------------------------

CROSSOVER_PROMPT = """\
You are an expert quantitative factor engineer specialized in **factor \
evolution and crossover design**.
{intro}

{complexity}

Your task is to generate a new alpha factor by **intelligently combining the \
following two parent factors**:

---
### Parent Factor 1:
<<parent factor 1>>
{parent_factor_1_code}
<</parent factor 1>>

---
### Parent Factor 2:
<<parent factor 2>>
{parent_factor_2_code}
<</parent factor 2>>

---
### Design objectives:
- Be creative and think deeply before taking the next step.
- Create a new alpha factor that combines the **core insights and signals** of \
both parent factors.
- Introduce meaningful **interactions** between the parent factors \
(non-linear, dynamic, temporal).
- The new factor should offer **potentially superior predictive power** and \
richer structure than either parent alone.
- Avoid simple additive combinations — instead, design **structurally novel** \
interactions.
- The new factor must remain interpretable and have clear financial intuition.

---
{requirements}

### Factor Design Guidance:
- Focus on capturing the essential intuition of the assigned theme.
- Ensure the logic is interpretable, robust, and implementable in a few steps.
- Prefer clean, generalizable formulas over highly engineered constructs.
- Each factor should be expressible in a short formula or <= 5 logical steps.
- Balance simplicity with predictive potential: avoid trivial duplication, but \
also avoid unnecessary complexity.

---
{extra_guidance}

---
{libraries}

---
{output_format}"""

MUTATION_PROMPT = """\
You are an expert quantitative factor engineer specialized in **factor \
evolution and mutation design**.
{intro}

{complexity}

Your task is to generate a new alpha factor by **slightly modifying the \
parent factor below to introduce variability**, while keeping its core \
hypothesis recognisable.

---
### Parent Factor:
<<parent factor>>
{parent_factor_code}
<</parent factor>>

---
### Design objectives:
- Change one meaningful aspect: the window, the normalisation, the transform, \
or the conditioning variable.
- Do not simply rename or re-scale the parent.
- Do not stack extra transforms on top of the parent; substitute rather than \
accumulate.
- The mutation must remain interpretable and have clear financial intuition.

---
{requirements}

---
{extra_guidance}

---
{libraries}

---
{output_format}"""


# --- Section 3.5: adaptive generation feedback ----------------------------

SUMMARISE_PROMPT = """\
Summarise, in the Observation -> Cause -> Fix form, why each of the following \
alpha factors performed as it did. One mini-chain per factor, at most three \
short lines each.

{samples}

Respond with the mini-chains only, as plain text."""

NO_FEEDBACK = "(no prior generations yet)"
