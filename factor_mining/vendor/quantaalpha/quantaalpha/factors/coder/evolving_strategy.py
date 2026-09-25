from __future__ import annotations

import json
from pathlib import Path
import re
from jinja2 import Environment, StrictUndefined

from quantaalpha.coder.costeer.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from quantaalpha.coder.costeer.knowledge_management import (
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.factors.coder.factor import FactorFBWorkspace, FactorTask
from quantaalpha.core.prompts import Prompts
from quantaalpha.core.template import CodeTemplate
from quantaalpha.llm.config import LLM_SETTINGS
from quantaalpha.llm.client import APIBackend
from quantaalpha.core.utils import multiprocessing_wrapper
from quantaalpha.core.conf import RD_AGENT_SETTINGS

code_template = CodeTemplate(template_path=Path(__file__).parent / "template.jinjia2")
implement_prompts = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

class FactorMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False


    def error_summary(
        self,
        target_task: FactorTask,
        queried_former_failed_knowledge_to_render: list,
        queried_similar_error_knowledge_to_render: list,
    ) -> str:
        error_summary_system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(implement_prompts["evolving_strategy_error_summary_v2_system"])
            .render(
                scenario=self.scen.get_scenario_all_desc(target_task),
                factor_information_str=target_task.get_task_information(),
                code_and_feedback=queried_former_failed_knowledge_to_render[-1].get_implementation_and_feedback_str(),
            )
            .strip("\n")
        )
        for _ in range(10):  # max attempt to reduce the length of error_summary_user_prompt
            error_summary_user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(implement_prompts["evolving_strategy_error_summary_v2_user"])
                .render(
                    queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                )
                .strip("\n")
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt
                )
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        error_summary_critics = APIBackend(
            use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
        ).build_messages_and_create_chat_completion(
            user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt, json_mode=False
        )
        return error_summary_critics

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
        )  # A list, [success task implement knowledge]

        if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2):
            queried_similar_error_knowledge = (
                queried_knowledge.task_to_similar_error_successful_knowledge[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )  # A dict, {{error_type:[[error_imp_knowledge, success_imp_knowledge],...]},...}
        else:
            queried_similar_error_knowledge = {}

        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][0]
            if queried_knowledge is not None
            else []
        )

        queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

        latest_attempt_to_latest_successful_execution = queried_knowledge.task_to_former_failed_traces[
            target_factor_task_information
        ][1]

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(
                implement_prompts["evolving_strategy_factor_implementation_v1_system"],
            )
            .render(
                scenario=self.scen.get_scenario_all_desc(target_task, filtered_tag="feature"),
                queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
            )
        )
        queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
        queried_similar_error_knowledge_to_render = queried_similar_error_knowledge
        for _ in range(10):
            # Optional error summary
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
            similar_successful_factor_description = ""
            similar_successful_expression = ""
            if len(queried_similar_successful_knowledge_to_render) > 0:
                similar_successful_factor_description = queried_similar_successful_knowledge_to_render[0].target_task.get_task_description()
                similar_successful_expression = self.extract_expr(queried_similar_successful_knowledge_to_render[0].implementation.code)
            
            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    implement_prompts["evolving_strategy_factor_implementation_v2_user"],
                )
                .render(
                    # factor_information_str=target_factor_task_information,
                    # queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                    # queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                    # error_summary_critics=error_summary_critics,
                    # latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
                    factor_information_str=target_task.get_task_description(),
                    queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                    error_summary_critics=error_summary_critics,
                    similar_successful_factor_description=similar_successful_factor_description,
                    similar_successful_expression=similar_successful_expression,
                    latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
                )
                .strip("\n")
            )
            if (
                APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_former_failed_knowledge_to_render) > 1:
                queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
            elif len(queried_similar_successful_knowledge_to_render) > len(
                queried_similar_error_knowledge_to_render,
            ):
                queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[:-1]
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        for _ in range(10):
            try:
                code = json.loads(
                    APIBackend(
                        use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
                    ).build_messages_and_create_chat_completion(
                        user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True
                    )
                )["code"]
                return code
            except json.decoder.JSONDecodeError:
                pass
        else:
            return ""  # return empty code if failed to get code after 10 attempts

    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_code(**{"factor.py": code_list[index]})
        return evo



qa_implement_prompts = Prompts(file_path=Path(__file__).parent / "qa_prompts.yaml")
class FactorParsingStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False

    def extract_expr(self, code_str: str) -> str:
        """Extract expr from code (expr = \"...\" or expr = '...')."""
        pattern = r'expr\s*=\s*["\']([^"\']*)["\']'
        match = re.search(pattern, code_str)
        if match:
            return match.group(1)
        else:
            return ""


    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
    ) -> str:
        """Generate code for one factor task. First run: template; on error: give LLM feedback and cases."""
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
            )  # A dict, {{error_type:[[error_imp_knowledge, success_imp_knowledge],...]},...}
        else:
            queried_similar_error_knowledge = {}

        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][0]
            if queried_knowledge is not None
            else []
        )

        queried_former_failed_knowledge_to_render = queried_former_failed_knowledge
        
        if len(queried_former_failed_knowledge) == 0:
            rendered_code = code_template.render(
                expression=target_task.factor_expression, 
                factor_name=target_task.factor_name 
            )
            return rendered_code
        
        else:
            latest_attempt_to_latest_successful_execution = queried_knowledge.task_to_former_failed_traces[
                target_factor_task_information
            ][1]

            system_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    qa_implement_prompts["evolving_strategy_factor_implementation_v1_system"],
                )
                .render(
                    scenario=self.scen.get_scenario_all_desc(target_task, filtered_tag="feature"),
                    # former_expression=self.extract_expr(queried_former_failed_knowledge_to_render[-1].implementation.code),
                    # former_feedback=queried_former_failed_knowledge_to_render[-1].feedback,
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
                    
                similar_successful_factor_description = ""
                similar_successful_expression = ""
                if len(queried_similar_successful_knowledge_to_render) > 0:
                    similar_successful_factor_description = queried_similar_successful_knowledge_to_render[-1].target_task.get_task_description()
                    similar_successful_expression = self.extract_expr(queried_similar_successful_knowledge_to_render[-1].implementation.code)
                
                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(
                        qa_implement_prompts["evolving_strategy_factor_implementation_v2_user"],
                    )
                    .render(
                        factor_information_str=target_task.get_task_description(),
                        queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                        former_expression=self.extract_expr(queried_former_failed_knowledge_to_render[-1].implementation.code),
                        former_feedback=queried_former_failed_knowledge_to_render[-1].feedback,
                        error_summary_critics=error_summary_critics,
                        similar_successful_factor_description=similar_successful_factor_description,
                        similar_successful_expression=similar_successful_expression,
                        latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
                    )
                    .strip("\n")
                )

                if (
                    APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                    < LLM_SETTINGS.chat_token_limit
                ):
                    break
                elif len(queried_former_failed_knowledge_to_render) > 1:
                    # Reduce former failed cases
                    queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
                elif len(queried_similar_successful_knowledge_to_render) > len(
                    queried_similar_error_knowledge_to_render,
                ):
                    # Reduce success cases
                    queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[:-1]
                elif len(queried_similar_error_knowledge_to_render) > 0:
                    # Reduce error cases
                    queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
                    
            for _ in range(10):
                try:
                    # Call API for new expression
                    expr = json.loads(
                        APIBackend(
                            use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
                        ).build_messages_and_create_chat_completion(
                            user_prompt=user_prompt, system_prompt=system_prompt, json_mode=True, reasoning_flag=False
                        )
                    )["expr"]
                    
                    # Render code template with new expression
                    rendered_code = code_template.render(
                        expression=expr, 
                        factor_name=target_task.factor_name 
                    )
                    return rendered_code
                    
                except json.decoder.JSONDecodeError:
                    pass  # JSON parse failed, retry
    
    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_code(**{"factor.py": code_list[index]})
        return evo
    
    
    
class FactorRunningStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False


    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
    ) -> str:

        rendered_code = code_template.render(
            expression=target_task.factor_expression, 
            factor_name=target_task.factor_name 
        )
        return rendered_code
        
    
    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_code(**{"factor.py": code_list[index]})
        return evo
    
    
    def evolve(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        **kwargs,
    ) -> EvolvingItem:
        # Find tasks to evolve
        to_be_finished_task_index = []
        for index, target_task in enumerate(evo.sub_tasks):
            to_be_finished_task_index.append(index)

        result = multiprocessing_wrapper(
            [
                (self.implement_one_task, (evo.sub_tasks[target_index], queried_knowledge))
                for target_index in to_be_finished_task_index
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )
        code_list = [None for _ in range(len(evo.sub_tasks))]
        for index, target_index in enumerate(to_be_finished_task_index):
            code_list[target_index] = result[index]

        evo = self.assign_code_list_to_evo(code_list, evo)
        evo.corresponding_selection = to_be_finished_task_index

        return evo
