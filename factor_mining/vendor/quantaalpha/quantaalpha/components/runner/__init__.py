import pickle
import shutil
from pathlib import Path
from typing import Any, Tuple

from quantaalpha.core.developer import Developer
from quantaalpha.core.experiment import ASpecificExp, Experiment
from quantaalpha.llm.client import md5_hash


class CachedRunner(Developer[ASpecificExp]):
    def get_cache_key(self, exp: Experiment, **kwargs) -> str:
        all_tasks = []
        for based_exp in exp.based_experiments:
            all_tasks.extend(based_exp.sub_tasks)
        all_tasks.extend(exp.sub_tasks)
        task_info_list = [task.get_task_information() for task in all_tasks]
        task_info_str = "\n".join(task_info_list)
        return md5_hash(task_info_str)

    def assign_cached_result(self, exp: Experiment, cached_res: Experiment) -> Experiment:
        """
        Assign cached experiment result to current exp.
        Ensures all experiments have running_info (may be missing after pickle deserialize).
        """
        try:
            from rdagent.core.experiment import RunningInfo
        except ImportError:
            try:
                from quantaalpha.core.experiment import RunningInfo
            except ImportError:
                RunningInfo = None
        
        def _ensure_running_info(exp_obj):
            """Ensure exp has running_info."""
            if RunningInfo is not None:
                if not hasattr(exp_obj, 'running_info') or exp_obj.running_info is None:
                    exp_obj.running_info = RunningInfo()
        
        _ensure_running_info(cached_res)
        if cached_res.based_experiments:
            for based_exp in cached_res.based_experiments:
                _ensure_running_info(based_exp)
        
        _ensure_running_info(exp)
        if exp.based_experiments:
            for based_exp in exp.based_experiments:
                _ensure_running_info(based_exp)
        
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1].result = cached_res.based_experiments[-1].result
        exp.result = cached_res.result
        return exp
