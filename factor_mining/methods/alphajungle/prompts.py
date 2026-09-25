"""LLM agent prompts, transcribed from the paper's Appendix K.

Figures 15-18 give these prompts in full, including their slot names
(``{available_fields}``, ``{freq_subtrees}``, ...). They are reproduced here as
closely as a plain-text rendering allows; the paper's figures are prose with
bullet lists, so the bullets become dashes and numbered lists keep their
numbers. Where a slot's contents are described but not shown (the operator
catalog, the window range), the value comes from `grammar.py`.

The refinement-suggestion prompt (the step that produces ``d_{s,i*}`` in
Eq. 4) is *described* in Section 3 and Appendix D but not printed as a figure;
`SUGGESTION_PROMPT` is written from that description and is flagged as ours in
METHOD.md.
"""

from __future__ import annotations

# Figure 15 — Alpha Portrait Generation Prompt.
PORTRAIT_PROMPT = """\
Task Description:
You are a quantitative finance expert specializing in factor-based investing. \
Please design an alpha factor used in investment strategies according to the \
following requirements, and then provide the content of the alpha in the \
required format.

Available Data Fields:
The following data fields are available for use:
{available_fields}

Available Operators:
The following operators are available for use:
{available_operators}

Alpha Requirements:
1. The alpha value should be dimensionless (unitless).
2. The alpha should incorporate at least two distinct operations from the \
"Available Operators" list to ensure it has sufficient complexity. Avoid \
creating overly simplistic alphas.
3. All look-back windows and other numerical parameters used in the alpha \
calculation MUST be represented as named parameters in the pseudo-code. These \
parameter names MUST follow Python naming conventions (e.g., lookback_period, \
volatility_window, smoothing_factor).
4. The alpha should have NO MORE than 3 parameters in total.
5. The pseudo-code should represent the alpha calculation step-by-step, using \
only the "Available Operators" and clearly defined parameters. Each line in \
the pseudo-code should represent a single operation.
6. Use descriptive variable names in the pseudo-code that clearly indicate the \
data they represent.
7. When designing alpha expressions, try to avoid including the following \
sub-expressions:
{freq_subtrees}

Formatting Requirements:
The output must be in JSON format with three key-value pairs:
1. "name": A short, descriptive name for the alpha (following Python variable \
naming style, e.g., price_volatility_ratio).
2. "description": A concise explanation of the alpha's purpose or what it \
measures. Avoid overly technical language. Focus on the intuition behind the \
alpha.
3. "pseudo_code": A list of strings, where each string is a line of simplified \
pseudo-code representing a single operation in the alpha calculation. Each \
line should follow the format: variable_name = op_name(input=[input1, input2, \
...], param=[param1, param2, ...]), where:
- variable_name is the output variable of the operation.
- op_name is the name of one of the "Available Operators".
- input1, input2, ... are input variables (either from "Available Data Fields" \
or previously calculated variables, cannot be of a numeric type).
- param1, param2, ... are parameter names defined in the alpha requirements.

The format example is as follows:
{{
  "name": "volatility_adjusted_momentum",
  "description": "......",
  "pseudo_code": ["......"]
}}

Respond with JSON only."""


# Figure 16 — Alpha Formula Generation Prompt.
FORMULA_PROMPT = """\
Task Description:
Please design a quantitative investment alpha expression according to the \
following requirements.

Available Data Fields:
- The following data fields are available for use: {available_fields}

Available Operators:
- The following operators are available for use: {available_operators}

Alpha Requirements:
- {alpha_portrait_prompt}

Formatting Requirements:
1. Provide the output in JSON format.
2. The JSON object should contain two fields: "formula", and "arguments".
- "formula": Represents the mathematical expression for calculating the alpha.
- "arguments": Represents the configurable parameters of the alpha.
3. "formula" is a list of dictionaries. Each dictionary represents a single \
operation and must contain four keys: "name", "param", "input", and "output".
- "name": The operator's name (a string), which MUST be one of the operators \
provided in the "Available Operators" section.
- "param": A list of strings, representing the parameter names for the \
operator. These parameter names MUST be used as keys in the "arguments" \
section.
- "input": A list of strings, representing the input variable names for the \
operator. These MUST be data fields from the "Available Data Fields" or output \
variables from previous operations in the "formula", cannot be of a numeric \
type.
- "output": A string, representing the output variable name for the operator. \
This output can be used as an input for subsequent operations.
4. "arguments" is a list of dictionaries. Each dictionary represents a set of \
parameter values for the alpha.
- The keys of each dictionary in "arguments" MUST correspond exactly to the \
parameter names defined in the "param" lists of the "formula".
- The values in each dictionary in "arguments" are the specific numerical \
values for the parameters.
5. You may include a maximum of 3 sets of parameters within the "arguments" \
field.
6. The parameter value that indicates the length of the lookback window (if \
applicable) must be within the {window_range} range.
7. Ensure that the alpha expression is both reasonable and computationally \
feasible.
8. Parameter names should be descriptive and follow Python naming conventions \
(e.g., window_size, lag_period, smoothing_factor). Avoid using single \
characters or numbers as parameter names.
9. Refer to the following example:
{{
  "formula": [
    {{"name": "Ma", "param": ["short_window"], "input": ["close"], \
"output": "fast_ma"}},
    {{"name": "Std", "param": ["long_window"], "input": ["close"], \
"output": "vol"}},
    {{"name": "Div", "param": [], "input": ["fast_ma", "vol"], \
"output": "alpha"}}
  ],
  "arguments": [
    {{"short_window": 5, "long_window": 20}},
    {{"short_window": 10, "long_window": 40}}
  ]
}}

Respond with JSON only."""


# Figure 17 — Alpha Overfitting Risk Assessment Prompt.
OVERFITTING_PROMPT = """\
Task: Critical Alpha Overfitting Risk Assessment

Critically evaluate the overfitting risk and generalization potential of the \
provided quantitative investment alpha, based on its expression and refinement \
history. Your assessment must focus on whether complexity and optimization \
appear justified or are likely signs of overfitting.

Input:
- Alpha Expression:
{alpha_formula}
- Refinement History:
{refinement_history}

Evaluation Criteria:
1. Justified Rationale vs. Complexity:
Critique: Is the complexity of the alpha expression plausibly justified by an \
inferred economic rationale, or does it seem arbitrary/excessive, suggesting \
fitting to noise?
2. Principled Development vs. Data Dredging:
Critique: Does the refinement history indicate hypothesis-driven improvements, \
or does it suggest excessive optimization and curve-fitting (e.g., frequent, \
unjustified parameter tweaks)?
3. Transparency vs. Opacity:
Critique: Is the alpha's logic reasonably interpretable despite its \
complexity, or is it opaque, potentially masking overfitting?

Scoring & Output:
- Assign a single Overfitting Risk Score from 0 to 10.
  - 10 = Very Low Risk (High confidence in generalization)
  - 0 = Very High Risk (Low confidence in generalization)
- Use the full 0-10 range to differentiate risk levels effectively.
- Provide a concise, one-sentence Justification explaining the score, citing \
the key factors from the criteria.
- Format the output as JSON, like the examples below:
{{"reason": "Complexity is justified by a strong rationale; principled \
refinement history suggests low risk.", "score": 9}}
{{"reason": "Plausible rationale, but some expression opacity and parameter \
tuning in history indicate moderate risk.", "score": 5}}
{{"reason": "High risk inferred from opaque expression lacking clear \
rationale, supported by history showing excessive tuning.", "score": 1}}

Respond with JSON only."""


# Figure 18 — Alpha Refinement Prompt.
REFINEMENT_PROMPT = """\
Task Description:
There is an alpha factor used in quantitative investment to predict asset \
price trends. Please improve it according to the following suggestions and \
provide the improved alpha expression.

Available Data Fields:
The following data fields are available for use: {available_fields}

Available Operators:
The following operators are available for use:
{available_operators}

Alpha Suggestions:
1. The alpha value should be dimensionless (unitless).
2. All look-back windows and other numerical parameters used in the alpha \
calculation MUST be represented as named parameters in the pseudo-code. These \
parameter names MUST follow Python naming conventions (e.g., lookback_period, \
volatility_window, smoothing_factor).
3. The alpha should have NO MORE than 3 parameters in total.
4. The pseudo-code should represent the alpha calculation step-by-step, using \
only the "Available Operators" and clearly defined parameters. Each line in \
the pseudo-code should represent a single operation.
5. Use descriptive variable names in the pseudo-code that clearly indicate the \
data they represent.
6. When designing alpha expressions, try to avoid including the following \
sub-expressions: {freq_subtrees}

Original alpha expression:
{origin_alpha_formula}

Refinement suggestions:
NOTE: The following improvement suggestions do not need to be all adopted; \
they just need to be considered and reasonable ones selected for adoption.
{refinement_suggestions}

Formatting Requirements:
The output must be in JSON format with three key-value pairs:
1. "name": A short, descriptive name for the alpha (following Python variable \
naming style, e.g., price_volatility_ratio).
2. "description": A concise explanation of the alpha's purpose or what it \
measures. Avoid overly technical language. Focus on the intuition behind the \
alpha.
3. "pseudo_code": A list of strings, where each string is a line of simplified \
pseudo-code representing a single operation in the alpha calculation, in the \
format variable_name = op_name(input=[...], param=[...]).

Respond with JSON only."""


# Not a paper figure. Section 3 / Appendix D describe this step ("the LLM
# generates a textual refinement suggestion d_{s,i*} aimed at improving
# performance on that dimension ... framed as a few-shot learning task, where
# the context includes effective alphas from F_zoo"), but do not print the
# prompt. Written from that description — see METHOD.md.
SUGGESTION_PROMPT = """\
Task Description:
You are a quantitative finance expert. An alpha factor is being iteratively \
refined. Propose how to improve it along ONE specific evaluation dimension.

Target dimension: {dimension}
What this dimension measures: {dimension_description}

Current alpha:
{alpha_formula}
Rationale: {alpha_description}

Its current evaluation scores (0 = worst, 1 = best, relative to the existing \
repository of effective alphas):
{score_table}

Refinement history of this alpha and its neighbours in the search tree:
{refinement_history}

{examples_block}

Requirements:
- Propose a conceptual change targeting the "{dimension}" dimension \
specifically. Do not restate the existing alpha.
- Do not repeat a refinement that the history shows has already been tried.
- Keep the economic rationale explicit: say what market behaviour the change \
is meant to capture.
- Avoid relying on these over-used structural motifs:
{freq_subtrees}

Formatting Requirements:
Respond with JSON only, with two key-value pairs:
1. "suggestion": one or two sentences describing the conceptual refinement.
2. "rationale": one sentence on why this should improve the \
"{dimension}" dimension."""


DIMENSION_DESCRIPTIONS = {
    "effectiveness": (
        "The alpha's core predictive power, measured by RankIC against future returns."
    ),
    "stability": (
        "The consistency of the alpha's predictive performance over time, measured by "
        "RankIR. An alpha that performs well only in specific market regimes is less "
        "reliable."
    ),
    "turnover": (
        "The trading cost associated with the alpha, measured by the average daily "
        "change in its portfolio holdings. High turnover implies frequent rebalancing, "
        "which erodes profits. The goal is to keep turnover within a desirable, low "
        "range."
    ),
    "diversity": (
        "The novelty the alpha contributes to the existing repository. A valuable alpha "
        "provides predictive signals that are not redundant with those already "
        "discovered."
    ),
    "overfitting": (
        "The risk that the alpha's complexity, parameter count, or refinement history "
        "reflects fitting to noise rather than a genuine economic effect."
    ),
}
