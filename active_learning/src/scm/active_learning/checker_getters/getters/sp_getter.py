from __future__ import annotations

from typing import ClassVar, List, Literal, Optional, Tuple, Type

from scm.moliterate import ConcatChemDataSet
from scm.moliterate.filters import RandomFilter, UnionFilters
from scm.moliterate.partition_criteria import MetadataCriterion, UnionPartitionCriterion

from scm.active_learning.checker_getters.getters.core import Getter
from scm.active_learning.results import (
    EngineCheckResult,
    FrameGetterResult,
    SPLabellingResults,
)
from scm.active_learning.tasks import SPLabellerData, Task


class SPGetter(Getter[SPLabellingResults]):
    type: Literal["SPGetter"] = "SPGetter"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (SPLabellerData,)
    getter_id: str = "SPGetter"
    filters_converged_sp: UnionFilters = RandomFilter()
    filters_failed_sp: Optional[UnionFilters] = None
    failure_checker: UnionPartitionCriterion = MetadataCriterion(metadata_keys=["FAILURE"])
    converged_failed_keys: Tuple[str, str] = ("True", "False")

    def run(
        self,
        result: SPLabellingResults,
        checker_results: List[EngineCheckResult],
        **kwargs,
    ) -> FrameGetterResult:
        if self.filters_failed_sp is None:
            dataset = self.filters_converged_sp(result.dataset)
        else:
            failed_entries = self.failure_checker(result.dataset)
            groups = self.failure_checker.group_by(failed_entries)
            ds1 = self.filters_converged_sp(result.dataset.subset(indices=groups[self.converged_failed_keys[0]]))
            ds2 = self.filters_failed_sp(result.dataset.subset(indices=groups[self.converged_failed_keys[1]]))
            dataset = ConcatChemDataSet(data_source=[ds1, ds2])
        return FrameGetterResult(
            engine_id=result.from_engine.engine_id,
            task_id=result.from_task.task_id,
            system_id=result.from_task.system_id,
            checker_id=checker_results[-1].checker_id,
            getter_id=self.getter_id,
            dataset=dataset,
        )
