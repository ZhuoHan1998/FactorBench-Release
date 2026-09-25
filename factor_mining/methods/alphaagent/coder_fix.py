"""Official-bug fix: AlphaAgent's expression-repair crash on empty knowledge.

``FactorParsingStrategy.implement_one_task`` (vendored @1da96e9) indexes
``queried_similar_successful_knowledge_to_render[-1]`` unconditionally when
repairing a failed factor. Until at least one factor has ever succeeded, that
list is empty, so a run whose FIRST factors all fail crashes with IndexError
— the exact round-1 scenario. Notably, the official prompt template already
guards the variables with ``{% if similar_successful_factor_description is
not none %}``; only the Python render call forgot the guard.

This module carries a corrected copy of that one method (verbatim from the
pinned commit, minus translated comments, plus the two guards) in a subclass,
injected via the official ``QLIB_FACTOR_CODER`` setting. No vendored file is
modified.
"""

from __future__ import annotations

import json

from alphaagent.components.coder.CoSTEER import CoSTEER
from alphaagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from alphaagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from alphaagent.components.coder.factor_coder import evolving_strategy as _es
from alphaagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from alphaagent.components.coder.factor_coder.evaluators import FactorEvaluatorForCoder
from alphaagent.components.coder.factor_coder.factor import FactorTask
from alphaagent.core.scenario import Scenario
from alphaagent.oai.llm_conf import LLM_SETTINGS
from alphaagent.oai.llm_utils import APIBackend
from jinja2 import Environment, StrictUndefined


class PatchedFactorParsingStrategy(_es.FactorParsingStrategy):
    """implement_one_task copied from the pinned vendor commit with guards."""

    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
    ) -> str:
        target_factor_task_information = target_task.get_task_information()

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_factor_task_information]
            if queried_knowledge is not None
            else []
        )
        if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2):
            queried_similar_error_knowledge = (
                queried_knowledge.task_to_similar_error_successful_knowledge[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )
        else:
            queried_similar_error_knowledge = {}

        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][0]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

        # First attempt: render the expression template directly (no LLM).
        if len(queried_former_failed_knowledge) == 0:
            return _es.code_template.render(
                expression=target_task.factor_expression,
                factor_name=target_task.factor_name,
            )

        task_traces = queried_knowledge.task_to_former_failed_traces
        latest_attempt_to_latest_successful_execution = task_traces[
            target_factor_task_information
        ][1]

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(
                _es.alphaagent_implement_prompts["evolving_strategy_factor_implementation_v1_system"],
            )
            .render(
                scenario=self.scen.get_scenario_all_desc(target_task, filtered_tag="feature"),
            )
        )
        queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
        queried_similar_error_knowledge_to_render = queried_similar_error_knowledge

        for _ in range(10):
            if (
                isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
                and FACTOR_COSTEER_SETTINGS.v2_error_summary
                and len(queried_similar_error_knowledge_to_render) != 0
                and len(queried_former_failed_knowledge_to_render) != 0
            ):
                error_summary_critics = self.error_summary(
                    target_task,
                    queried_former_failed_knowledge_to_render,
                    queried_similar_error_knowledge_to_render,
                )
            else:
                error_summary_critics = None

            # FIX vs official: the successful-knowledge list is empty until any
            # factor has ever succeeded; official code indexed [-1] blindly.
            # The prompt template already handles None for both variables.
            has_success = len(queried_similar_successful_knowledge_to_render) > 0
            similar_successful_description = (
                queried_similar_successful_knowledge_to_render[-1].target_task.get_task_description()
                if has_success
                else None
            )
            similar_successful_expression = (
                self.extract_expr(queried_similar_successful_knowledge_to_render[-1].implementation.code)
                if has_success
                else None
            )

            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    _es.alphaagent_implement_prompts["evolving_strategy_factor_implementation_v2_user"],
                )
                .render(
                    factor_information_str=target_task.get_task_description(),
                    queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                    former_expression=self.extract_expr(
                        queried_former_failed_knowledge_to_render[-1].implementation.code
                    ),
                    former_feedback=queried_former_failed_knowledge_to_render[-1].feedback,
                    error_summary_critics=error_summary_critics,
                    similar_successful_factor_description=similar_successful_description,
                    similar_successful_expression=similar_successful_expression,
                    latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
                )
                .strip("\n")
            )

            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt, system_prompt=system_prompt
                )
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_former_failed_knowledge_to_render) > 1:
                queried_former_failed_knowledge_to_render = (
                    queried_former_failed_knowledge_to_render[1:]
                )
            elif len(queried_similar_successful_knowledge_to_render) > len(
                queried_similar_error_knowledge_to_render,
            ):
                queried_similar_successful_knowledge_to_render = (
                    queried_similar_successful_knowledge_to_render[:-1]
                )
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = (
                    queried_similar_error_knowledge_to_render[:-1]
                )

        for _ in range(10):
            try:
                expr = json.loads(
                    APIBackend(
                        use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
                    ).build_messages_and_create_chat_completion(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                        json_mode=True,
                        reasoning_flag=False,
                    )
                )["expr"]
                return _es.code_template.render(
                    expression=expr,
                    factor_name=target_task.factor_name,
                )
            except json.decoder.JSONDecodeError:
                pass


class PatchedFactorParser(CoSTEER):
    """Official FactorParser wiring, with the patched parsing strategy."""

    def __init__(self, scen: Scenario, *args, **kwargs) -> None:
        setting = FACTOR_COSTEER_SETTINGS
        eva = CoSTEERMultiEvaluator(FactorEvaluatorForCoder(scen=scen), scen=scen)
        es = PatchedFactorParsingStrategy(scen=scen, settings=FACTOR_COSTEER_SETTINGS)
        super().__init__(
            *args, settings=setting, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs
        )
