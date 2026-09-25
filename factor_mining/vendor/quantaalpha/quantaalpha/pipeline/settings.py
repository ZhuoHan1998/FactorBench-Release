"""
QuantaAlpha pipeline settings.

Defines class-path configuration for pipeline components.
Components are loaded dynamically via string class paths for flexibility.
"""

from quantaalpha.core.conf import ExtendedBaseSettings, ExtendedSettingsConfigDict


# =============================================================================
# Base setting classes
# =============================================================================

class BasePropSetting(ExtendedBaseSettings):
    """Common base for RD Loop configuration."""

    scen: str = ""
    knowledge_base: str = ""
    knowledge_base_path: str = ""
    hypothesis_gen: str = ""
    hypothesis2experiment: str = ""
    coder: str = ""
    runner: str = ""
    summarizer: str = ""
    evolving_n: int = 10


class BaseFacSetting(ExtendedBaseSettings):
    """Common base for Alpha Agent Loop configuration."""

    scen: str = ""
    knowledge_base: str = ""
    knowledge_base_path: str = ""
    hypothesis_gen: str = ""
    construction: str = ""
    calculation: str = ""
    coder: str = ""
    runner: str = ""
    summarizer: str = ""
    evolving_n: int = 10


# =============================================================================
# Factor mining settings (main experiment)
# =============================================================================

class AlphaAgentFactorBasePropSetting(BasePropSetting):
    """Main experiment: LLM-driven factor mining."""
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    scen: str = "quantaalpha.factors.experiment.QlibAlphaAgentScenario"
    hypothesis_gen: str = "quantaalpha.factors.proposal.AlphaAgentHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.factors.proposal.AlphaAgentHypothesis2FactorExpression"
    coder: str = "quantaalpha.factors.qlib_coder.QlibFactorParser"
    runner: str = "quantaalpha.factors.runner.QlibFactorRunner"
    summarizer: str = "quantaalpha.factors.feedback.AlphaAgentQlibFactorHypothesisExperiment2Feedback"
    evolving_n: int = 5


class FactorBasePropSetting(BasePropSetting):
    """Basic factor experiment (traditional RD Loop mode)."""
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    scen: str = "quantaalpha.factors.experiment.QlibFactorScenario"
    hypothesis_gen: str = "quantaalpha.factors.proposal.QlibFactorHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.factors.proposal.QlibFactorHypothesis2Experiment"
    coder: str = "quantaalpha.factors.qlib_coder.QlibFactorCoSTEER"
    runner: str = "quantaalpha.factors.runner.QlibFactorRunner"
    summarizer: str = "quantaalpha.factors.feedback.QlibFactorHypothesisExperiment2Feedback"
    evolving_n: int = 10


class FactorBackTestBasePropSetting(BasePropSetting):
    """Factor backtest mode."""
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    scen: str = "quantaalpha.factors.experiment.QlibAlphaAgentScenario"
    hypothesis_gen: str = "quantaalpha.factors.proposal.EmptyHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.factors.proposal.BacktestHypothesis2FactorExpression"
    coder: str = "quantaalpha.factors.qlib_coder.QlibFactorCoder"
    runner: str = "quantaalpha.factors.runner.QlibFactorRunner"
    summarizer: str = "quantaalpha.factors.feedback.QlibFactorHypothesisExperiment2Feedback"
    evolving_n: int = 1


class FactorFromReportPropSetting(FactorBasePropSetting):
    """Factor extraction from research reports."""
    scen: str = "quantaalpha.factors.experiment.QlibFactorFromReportScenario"
    report_result_json_file_path: str = "git_ignore_folder/report_list.json"
    max_factors_per_exp: int = 10000
    is_report_limit_enabled: bool = False


# =============================================================================
# Model experiment settings (contrib, optional)
# =============================================================================

class ModelBasePropSetting(BasePropSetting):
    """Model experiment (extended feature)."""
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_MODEL_", protected_namespaces=())

    scen: str = "quantaalpha.contrib.model.experiment.QlibModelScenario"
    hypothesis_gen: str = "quantaalpha.contrib.model.proposal.QlibModelHypothesisGen"
    hypothesis2experiment: str = "quantaalpha.contrib.model.proposal.QlibModelHypothesis2Experiment"
    coder: str = "quantaalpha.contrib.model.qlib_coder.QlibModelCoSTEER"
    runner: str = "quantaalpha.contrib.model.runner.QlibModelRunner"
    summarizer: str = "quantaalpha.factors.feedback.QlibModelHypothesisExperiment2Feedback"
    evolving_n: int = 10


# =============================================================================
# Singleton instances (global)
# =============================================================================

ALPHA_AGENT_FACTOR_PROP_SETTING = AlphaAgentFactorBasePropSetting()
FACTOR_PROP_SETTING = FactorBasePropSetting()
FACTOR_BACK_TEST_PROP_SETTING = FactorBackTestBasePropSetting()
FACTOR_FROM_REPORT_PROP_SETTING = FactorFromReportPropSetting()
MODEL_PROP_SETTING = ModelBasePropSetting()
