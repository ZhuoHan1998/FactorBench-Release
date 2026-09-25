"""
Factor Consistency Checker: semantic consistency between hypothesis, description, formulation, expression, code.
"""

import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from jinja2 import Environment, StrictUndefined

from quantaalpha.core.prompts import Prompts
from quantaalpha.llm.client import APIBackend, robust_json_parse
from quantaalpha.log import logger

consistency_prompts = Prompts(file_path=Path(__file__).parent / "consistency_prompts.yaml")


@dataclass
class ConsistencyCheckResult:
    """Result of consistency check (hypothesis->description->formulation->expression)."""
    is_consistent: bool
    hypothesis_to_description: str
    description_to_formulation: str
    formulation_to_expression: str
    overall_feedback: str
    corrected_expression: Optional[str] = None
    corrected_description: Optional[str] = None
    severity: str = "none"  # none, minor, major, critical
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "hypothesis_to_description": self.hypothesis_to_description,
            "description_to_formulation": self.description_to_formulation,
            "formulation_to_expression": self.formulation_to_expression,
            "overall_feedback": self.overall_feedback,
            "corrected_expression": self.corrected_expression,
            "corrected_description": self.corrected_description,
            "severity": self.severity
        }


class FactorConsistencyChecker:
    """Checks logical consistency between hypothesis, description, formulation, expression; corrects if needed."""

    def __init__(
        self,
        scen=None,
        max_correction_attempts: int = 3,
        enabled: bool = True,
        strict_mode: bool = False
    ):
        """scen: scenario; max_correction_attempts: max correction tries; enabled; strict_mode (reject on any inconsistency)."""
        self.scen = scen
        self.max_correction_attempts = max_correction_attempts
        self.enabled = enabled
        self.strict_mode = strict_mode
    
    def check_consistency(
        self,
        hypothesis: str,
        factor_name: str,
        factor_description: str,
        factor_formulation: str,
        factor_expression: str,
        variables: Dict[str, str] = None
    ) -> ConsistencyCheckResult:
        """Check consistency between hypothesis, description, formulation, expression; return result or corrected fields."""
        if not self.enabled:
            return ConsistencyCheckResult(
                is_consistent=True,
                hypothesis_to_description="Consistency check disabled",
                description_to_formulation="Consistency check disabled",
                formulation_to_expression="Consistency check disabled",
                overall_feedback="Consistency check is disabled, skipping.",
                severity="none"
            )
        
        logger.info(f"Starting consistency check: {factor_name}")
        
        try:
            system_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(consistency_prompts["consistency_check_system"])
                .render()
            )
            
            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(consistency_prompts["consistency_check_user"])
                .render(
                    hypothesis=hypothesis,
                    factor_name=factor_name,
                    factor_description=factor_description,
                    factor_formulation=factor_formulation,
                    factor_expression=factor_expression,
                    variables=variables or {}
                )
            )
            
            response = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True
            )
            
            result_dict = robust_json_parse(response)
            
            is_consistent = result_dict.get("is_consistent", False)
            severity = result_dict.get("severity", "none")
            
            result = ConsistencyCheckResult(
                is_consistent=is_consistent,
                hypothesis_to_description=result_dict.get("hypothesis_to_description", ""),
                description_to_formulation=result_dict.get("description_to_formulation", ""),
                formulation_to_expression=result_dict.get("formulation_to_expression", ""),
                overall_feedback=result_dict.get("overall_feedback", ""),
                corrected_expression=result_dict.get("corrected_expression"),
                corrected_description=result_dict.get("corrected_description"),
                severity=severity
            )
            
            if is_consistent:
                logger.info(f"Consistency check passed: {factor_name}")
            else:
                logger.warning(f"Consistency check failed: {factor_name}, severity: {severity}")
                logger.warning(f"Feedback: {result.overall_feedback}")
            
            return result
        
        except Exception as e:
            logger.error(f"Consistency check error: {e}")
            return ConsistencyCheckResult(
                is_consistent=True,
                hypothesis_to_description=f"Error during check: {str(e)}",
                description_to_formulation="",
                formulation_to_expression="",
                overall_feedback=f"Consistency check failed with error: {str(e)}. Skipping check.",
                severity="none"
            )
    
    def check_and_correct(
        self,
        hypothesis: str,
        factor_name: str,
        factor_description: str,
        factor_formulation: str,
        factor_expression: str,
        variables: Dict[str, str] = None
    ) -> Tuple[ConsistencyCheckResult, str, str]:
        """Check consistency and attempt correction. Returns (result, final_expr, final_desc)."""
        current_expression = factor_expression
        current_description = factor_description
        
        for attempt in range(self.max_correction_attempts):
            result = self.check_consistency(
                hypothesis=hypothesis,
                factor_name=factor_name,
                factor_description=current_description,
                factor_formulation=factor_formulation,
                factor_expression=current_expression,
                variables=variables
            )
            
            if result.is_consistent:
                return result, current_expression, current_description
            
            if self.strict_mode:
                logger.warning(f"Strict mode: factor {factor_name} failed, no correction")
                return result, current_expression, current_description
            
            if result.corrected_expression and result.corrected_expression != current_expression:
                logger.info(f"Attempting expression correction ({attempt + 1}/{self.max_correction_attempts})")
                logger.info(f"Original: {current_expression}")
                logger.info(f"Corrected: {result.corrected_expression}")
                current_expression = result.corrected_expression
            elif result.corrected_description and result.corrected_description != current_description:
                logger.info(f"Attempting description correction ({attempt + 1}/{self.max_correction_attempts})")
                current_description = result.corrected_description
            else:
                logger.warning(f"Cannot correct factor {factor_name}, giving up")
                break
        
        final_result = self.check_consistency(
            hypothesis=hypothesis,
            factor_name=factor_name,
            factor_description=current_description,
            factor_formulation=factor_formulation,
            factor_expression=current_expression,
            variables=variables
        )
        
        return final_result, current_expression, current_description
    
    def batch_check(
        self,
        hypothesis: str,
        factors: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], ConsistencyCheckResult]]:
        """Batch check consistency for multiple factors."""
        results = []
        
        for factor in factors:
            result, corrected_expr, corrected_desc = self.check_and_correct(
                hypothesis=hypothesis,
                factor_name=factor.get("name", "Unknown"),
                factor_description=factor.get("description", ""),
                factor_formulation=factor.get("formulation", ""),
                factor_expression=factor.get("expression", ""),
                variables=factor.get("variables", {})
            )
            
            updated_factor = factor.copy()
            updated_factor["expression"] = corrected_expr
            updated_factor["description"] = corrected_desc
            updated_factor["consistency_check"] = result.to_dict()
            
            results.append((updated_factor, result))
        
        return results
    
    def should_proceed_to_backtest(self, result: ConsistencyCheckResult) -> bool:
        """Whether to proceed to backtest based on consistency result."""
        if not self.enabled:
            return True
        
        if result.is_consistent:
            return True
        
        if self.strict_mode:
            return False
        
        if result.severity in ["none", "minor"]:
            return True
        
        return False


class ComplexityChecker:
    """Factor complexity checker: validates expression complexity."""
    
    def __init__(
        self,
        enabled: bool = True,
        symbol_length_threshold: int = 250,
        base_features_threshold: int = 6,
        free_args_ratio_threshold: float = 0.5
    ):
        """Args: enabled, symbol_length_threshold, base_features_threshold, free_args_ratio_threshold."""
        self.enabled = enabled
        self.symbol_length_threshold = symbol_length_threshold
        self.base_features_threshold = base_features_threshold
        self.free_args_ratio_threshold = free_args_ratio_threshold
    
    def check(self, expression: str) -> Tuple[bool, str]:
        """Check expression complexity. Returns (passed, feedback)."""
        if not self.enabled:
            return True, "Complexity check disabled"
        
        try:
            from quantaalpha.factors.coder.factor_ast import (
                calculate_symbol_length, 
                count_base_features,
                count_free_args,
                count_all_nodes
            )
            
            feedback_parts = []
            passed = True
            
            symbol_length = calculate_symbol_length(expression)
            if symbol_length > self.symbol_length_threshold:
                passed = False
                feedback_parts.append(
                    f"Symbol Length (SL) Check Failed: {symbol_length} > {self.symbol_length_threshold}. "
                    f"Expression is too complex and may lead to overfitting."
                )
            
            num_base_features = count_base_features(expression)
            if num_base_features > self.base_features_threshold:
                passed = False
                feedback_parts.append(
                    f"Base Features (ER) Check Failed: {num_base_features} > {self.base_features_threshold}. "
                    f"Using too many raw features."
                )
            
            num_free_args = count_free_args(expression)
            num_all_nodes = count_all_nodes(expression)
            if num_all_nodes > 0:
                free_args_ratio = num_free_args / num_all_nodes
                if free_args_ratio > self.free_args_ratio_threshold:
                    passed = False
                    feedback_parts.append(
                        f"Free Args Ratio Check Failed: {free_args_ratio:.2%} > {self.free_args_ratio_threshold:.2%}. "
                        f"Factor is over-parameterized."
                    )
            
            if passed:
                return True, "Complexity check passed"
            else:
                return False, "\n".join(feedback_parts)
        
        except Exception as e:
            logger.warning(f"Complexity check failed with error: {e}")
            return True, f"Complexity check skipped due to error: {e}"


class RedundancyChecker:
    """Redundancy checker: detects duplication with existing factors."""
    
    def __init__(
        self,
        enabled: bool = True,
        duplication_threshold: int = 5,
        factor_zoo_path: str = None
    ):
        """Args: enabled, duplication_threshold, factor_zoo_path."""
        self.enabled = enabled
        self.duplication_threshold = duplication_threshold
        self.factor_zoo_path = factor_zoo_path
        self._factor_regulator = None
    
    @property
    def factor_regulator(self):
        """Lazy-load FactorRegulator."""
        if self._factor_regulator is None:
            from quantaalpha.factors.regulator.factor_regulator import FactorRegulator
            self._factor_regulator = FactorRegulator(
                factor_zoo_path=self.factor_zoo_path,
                duplication_threshold=self.duplication_threshold
            )
        return self._factor_regulator
    
    def check(self, expression: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Check expression redundancy. Returns (passed, feedback, details)."""
        if not self.enabled:
            return True, "Redundancy check disabled", {}
        
        try:
            if not self.factor_regulator.is_parsable(expression):
                return False, f"Expression cannot be parsed: {expression}", {}
            
            success, eval_dict = self.factor_regulator.evaluate(expression)
            if not success:
                return False, f"Failed to evaluate expression", {}
            
            duplicated_size = eval_dict.get('duplicated_subtree_size', 0)
            if duplicated_size > self.duplication_threshold:
                matched_alpha = eval_dict.get('matched_alpha', 'Unknown')
                duplicated_subtree = eval_dict.get('duplicated_subtree', '')
                return False, (
                    f"Redundancy Check Failed: Duplicated subtree size ({duplicated_size}) "
                    f"exceeds threshold ({self.duplication_threshold}). "
                    f"Matched with: {matched_alpha}. Duplicated subtree: {duplicated_subtree}"
                ), eval_dict
            
            return True, "Redundancy check passed", eval_dict
        
        except Exception as e:
            logger.warning(f"Redundancy check failed with error: {e}")
            return True, f"Redundancy check skipped due to error: {e}", {}


class FactorQualityGate:
    """Factor quality gate: integrates consistency/complexity/redundancy checks to decide if factor can proceed to backtest."""
    
    def __init__(
        self,
        consistency_checker: FactorConsistencyChecker = None,
        complexity_checker: ComplexityChecker = None,
        redundancy_checker: RedundancyChecker = None,
        consistency_enabled: bool = False,
        complexity_enabled: bool = True,
        redundancy_enabled: bool = True
    ):
        """Args: consistency_checker, complexity_checker, redundancy_checker, *_enabled flags."""
        self.consistency_checker = consistency_checker or FactorConsistencyChecker(enabled=consistency_enabled)
        self.complexity_checker = complexity_checker or ComplexityChecker(enabled=complexity_enabled)
        self.redundancy_checker = redundancy_checker or RedundancyChecker(enabled=redundancy_enabled)
        
        self.consistency_checker.enabled = consistency_enabled
        self.complexity_checker.enabled = complexity_enabled
        self.redundancy_checker.enabled = redundancy_enabled
    
    def evaluate(
        self,
        hypothesis: str,
        factor_name: str,
        factor_description: str,
        factor_formulation: str,
        factor_expression: str,
        variables: Dict[str, str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Evaluate if factor passes quality gate. Returns (passed, overall_feedback, results)."""
        results = {
            "consistency": None,
            "complexity": None,
            "redundancy": None,
            "corrected_expression": factor_expression,
            "corrected_description": factor_description
        }
        feedbacks = []
        all_passed = True
        
        if self.consistency_checker.enabled:
            consistency_result, corrected_expr, corrected_desc = self.consistency_checker.check_and_correct(
                hypothesis=hypothesis,
                factor_name=factor_name,
                factor_description=factor_description,
                factor_formulation=factor_formulation,
                factor_expression=factor_expression,
                variables=variables
            )
            results["consistency"] = consistency_result.to_dict()
            results["corrected_expression"] = corrected_expr
            results["corrected_description"] = corrected_desc
            
            if not self.consistency_checker.should_proceed_to_backtest(consistency_result):
                all_passed = False
                feedbacks.append(f"[Consistency] {consistency_result.overall_feedback}")
            
            factor_expression = corrected_expr
        
        if self.complexity_checker.enabled:
            complexity_passed, complexity_feedback = self.complexity_checker.check(factor_expression)
            results["complexity"] = {
                "passed": complexity_passed,
                "feedback": complexity_feedback
            }
            
            if not complexity_passed:
                all_passed = False
                feedbacks.append(f"[Complexity] {complexity_feedback}")
        
        if self.redundancy_checker.enabled:
            redundancy_passed, redundancy_feedback, redundancy_details = self.redundancy_checker.check(factor_expression)
            results["redundancy"] = {
                "passed": redundancy_passed,
                "feedback": redundancy_feedback,
                "details": redundancy_details
            }
            
            if not redundancy_passed:
                all_passed = False
                feedbacks.append(f"[Redundancy] {redundancy_feedback}")
        
        if all_passed:
            overall_feedback = f"Factor '{factor_name}' passed all quality gates."
            logger.info(overall_feedback)
        else:
            overall_feedback = f"Factor '{factor_name}' failed quality gates:\n" + "\n".join(feedbacks)
            logger.warning(overall_feedback)
        
        return all_passed, overall_feedback, results
