from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

if TYPE_CHECKING:
    from quantaalpha.core.evaluation import Evaluator
    from quantaalpha.core.evolving_framework import EvolvableSubjects

from quantaalpha.core.evaluation import Feedback
from quantaalpha.core.evolving_framework import EvolvingStrategy, EvoStep
from quantaalpha.log import logger


class EvoAgent(ABC):
    def __init__(self, max_loop: int, evolving_strategy: EvolvingStrategy) -> None:
        self.max_loop = max_loop
        self.evolving_strategy = evolving_strategy

    @abstractmethod
    def multistep_evolve(
        self,
        evo: EvolvableSubjects,
        eva: Evaluator | Feedback,
        filter_final_evo: bool = False,
    ) -> EvolvableSubjects: ...

    @abstractmethod
    def filter_evolvable_subjects_by_feedback(
        self,
        evo: EvolvableSubjects,
        feedback: Feedback | None,
    ) -> EvolvableSubjects: ...


class RAGEvoAgent(EvoAgent):
    def __init__(
        self,
        max_loop: int,
        evolving_strategy: EvolvingStrategy,
        rag: Any,
        with_knowledge: bool = False,
        with_feedback: bool = True,
        knowledge_self_gen: bool = False,
    ) -> None:
        super().__init__(max_loop, evolving_strategy)
        self.rag = rag
        self.evolving_trace: list[EvoStep] = []
        self.with_knowledge = with_knowledge
        self.with_feedback = with_feedback
        self.knowledge_self_gen = knowledge_self_gen

    def multistep_evolve(
        self,
        evo: EvolvableSubjects,
        eva: Evaluator | Feedback,
        filter_final_evo: bool = False,
    ) -> EvolvableSubjects:
        """Multi-step evolution: knowledge self-evolve, RAG query, evolve, pack, evaluate, update trace."""
        for loop_i in tqdm(range(self.max_loop), "Debugging"):
            if self.knowledge_self_gen and self.rag is not None:
                self.rag.generate_knowledge(self.evolving_trace)
                
            queried_knowledge = None
            if self.with_knowledge and self.rag is not None:
                queried_knowledge = self.rag.query(evo, self.evolving_trace)

            evo = self.evolving_strategy.evolve(
                evo=evo,
                evolving_trace=self.evolving_trace,
                queried_knowledge=queried_knowledge,
            )
            
            logger.log_object(evo.sub_workspace_list, tag="evolving code")  # type: ignore[attr-defined]
            if loop_i == 0:
                for sw in evo.sub_workspace_list:  # type: ignore[attr-defined]
                    logger.info(f"evolving code workspace: {sw}")

            es = EvoStep(evo, queried_knowledge)

            if self.with_feedback:
                es.feedback = (
                    eva
                    if isinstance(eva, Feedback)
                    else eva.evaluate(evo, queried_knowledge=queried_knowledge)  # type: ignore[arg-type, call-arg]
                )
                logger.log_object(es.feedback, tag="evolving feedback")

            self.evolving_trace.append(es)

        # If feedback enabled and filter requested, filter by last feedback
        if self.with_feedback and filter_final_evo:
            evo = self.filter_evolvable_subjects_by_feedback(evo, self.evolving_trace[-1].feedback)
        return evo

    def filter_evolvable_subjects_by_feedback(
        self,
        evo: EvolvableSubjects,
        feedback: Feedback | None,
    ) -> EvolvableSubjects:
        # Implementation of filter_evolvable_subjects_by_feedback method
        pass
