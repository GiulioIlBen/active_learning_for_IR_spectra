from __future__ import annotations

from typing import ClassVar, List, Literal, Tuple, Type, Union

from pydantic import model_validator
from scm.moliterate.analysis import PairwiseDatasetMetrics

from scm.active_learning.checker_getters.checkers.core import Checker
from scm.active_learning.results import EngineCheckResult, SPLabellingResults
from scm.active_learning.tasks import SPLabellerData, Task


class SPChecker(Checker[SPLabellingResults]):
    type: Literal["SPChecker"] = "SPChecker"
    supported_tasks: ClassVar[Tuple[Type[Task]]] = (SPLabellerData,)
    checker_id: str
    metrics: PairwiseDatasetMetrics
    n_failures_allowed: int = 0
    groupby_metadata: Union[List[str], Literal["retain_previous"]] = []
    collect_ids_from_groupinfo: bool = False

    @model_validator(mode="after")
    def _resolve_groupby_metadata(self):
        if self.groupby_metadata == "retain_previous":
            self.groupby_metadata = self.groupby_metadata_for_al
            self.collect_ids_from_groupinfo = True
        return self

    @property
    def groupby_metadata_for_al(self):
        return ["engine_id", "system_id", "task_id", "checker_id"]

    def run(self, result: SPLabellingResults, **kwargs) -> List[EngineCheckResult]:
        res = self.check_failed_sp(result)
        # res.extend(self.check_failed_sp(result))
        if len(res) > 0:
            return res
        if len(self.groupby_metadata) == 0:
            pairwise_results = self.metrics.compare_two(result.task.dataset, result.dataset)
        else:
            pairwise_results = self.metrics.compare_two_grouped(
                result.task.dataset,
                result.dataset,
                metadata_grouping=self.groupby_metadata,
            )
        extra_fields = (
            (lambda x: x.group_info) if self.collect_ids_from_groupinfo else (lambda x: self.collect_ids(result=result))
        )
        return [
            EngineCheckResult(
                **extra_fields(x),
                **x.model_dump(exclude={"group_info"}),
            )
            for x in pairwise_results
        ]

    def check_failed_sp(self, result: SPLabellingResults, **kwargs) -> List[EngineCheckResult]:
        if len(result.failed_simulations) > self.n_failures_allowed:
            return [
                EngineCheckResult(
                    **self.collect_ids(result=result),  # pyright: ignore[reportArgumentType]
                    property="failed_simulations",
                    metric="count",
                    value=len(result.failed_simulations),
                    target=self.n_failures_allowed,
                    n_entries=len(result.failed_simulations),
                    success="SP_simulations_failed",
                )
            ]
        return []
