"""
QuantaAlpha factor experiment module: Scenario and Experiment classes.
Uses project QlibFBWorkspace (no ProcessInf / pandas 1.5.x issues).
"""

from copy import deepcopy
from pathlib import Path

from rdagent.scenarios.qlib.experiment.factor_experiment import (  # type: ignore
    QlibFactorScenario,
    FactorExperiment,
    FactorTask,
    FactorFBWorkspace,
)
from rdagent.utils.agent.tpl import T

from quantaalpha.factors.workspace import QlibFBWorkspace
from rdagent.scenarios.qlib.experiment.factor_experiment import (
    QlibFactorExperiment as _OrigQlibFactorExperiment,
)


class QlibFactorExperiment(_OrigQlibFactorExperiment):
    """Override rdagent QlibFactorExperiment with project QlibFBWorkspace (correct config template)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        import rdagent.scenarios.qlib.experiment.factor_experiment as _fe_mod

        rdagent_template_path = Path(_fe_mod.__file__).parent / "factor_template"
        self.experiment_workspace = QlibFBWorkspace(
            template_folder_path=rdagent_template_path
        )


class QlibAlphaAgentScenario(QlibFactorScenario):
    """Scenario wrapper for AlphaAgent: accepts use_local; when True uses local get_data_folder_intro (no Docker)."""

    def __init__(self, use_local: bool = True, *args, **kwargs):
        from rdagent.core.scenario import Scenario
        from quantaalpha.factors.qlib_utils import get_data_folder_intro as local_get_data_folder_intro

        Scenario.__init__(self)
        tpl_prefix = "scenarios.qlib.experiment.prompts"

        self._background = deepcopy(
            T(f"{tpl_prefix}:qlib_factor_background").r(
                runtime_environment=self.get_runtime_environment(),
            )
        )
        self._source_data = deepcopy(local_get_data_folder_intro(use_local=use_local))
        self._output_format = deepcopy(T(f"{tpl_prefix}:qlib_factor_output_format").r())
        self._interface = deepcopy(T(f"{tpl_prefix}:qlib_factor_interface").r())
        self._strategy = deepcopy(T(f"{tpl_prefix}:qlib_factor_strategy").r())
        self._simulator = deepcopy(T(f"{tpl_prefix}:qlib_factor_simulator").r())
        self._rich_style_description = deepcopy(T(f"{tpl_prefix}:qlib_factor_rich_style_description").r())
        self._experiment_setting = deepcopy(T(f"{tpl_prefix}:qlib_factor_experiment_setting").r())
