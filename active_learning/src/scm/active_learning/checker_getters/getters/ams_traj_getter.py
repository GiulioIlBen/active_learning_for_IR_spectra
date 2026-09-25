from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, Field
from scm.moliterate import ConcreteInterfaces, load_dataset
from scm.moliterate.filters import LinearSteppedFilter, UnionFilters
from scm.moliterate.interfaces import InMemoryMolData
from typing_extensions import Annotated

from scm.active_learning.checker_getters.getters.core import Getter
from scm.active_learning.logging import log_level
from scm.active_learning.results import EngineCheckResult, FrameGetterResult, TaskResult
from scm.active_learning.results.task import AMSTaskResult
from scm.active_learning.tasks import AMSMDTask, AMSTask, Task

SliceTuple = Tuple[Optional[int], Optional[int], Optional[int]]

# Unofrtunatly this pattern is useful, because from the settings is clearer the logic of each step, hopefully.
# (uncertainty should be addded for example...)


class ConditionFilter(BaseModel):
    def run(self, ds: ConcreteInterfaces, checker_results: List[EngineCheckResult], **kwargs) -> ConcreteInterfaces:
        return ds


class RemoveUnreasonableTraj(ConditionFilter):
    type: Literal["RemoveUnreasonableTraj"] = "RemoveUnreasonableTraj"
    checker_results_query: Optional[str] = "property='trajectory',metric='reasonable_final_frame'"
    energy_change_frame: Literal["previous", "next_if_go_converged"] = "previous"

    def run(self, ds: ConcreteInterfaces, checker_results: List[EngineCheckResult], **kwargs) -> ConcreteInterfaces:
        if self.checker_results_query is not None:
            check_results_interesting = list(Getter.query_checker_results(self.checker_results_query, checker_results))
            if (
                len(check_results_interesting) == 1
                and isinstance(check_results_interesting[0].value, float)
                and check_results_interesting[0].success != "OK"
            ):
                trajectory_result = check_results_interesting[0]
                reasonable_final_frame = int(trajectory_result.value)
                stop = reasonable_final_frame + 1
                go_convergence_results = list(
                    Getter.query_checker_results(
                        "property='ConvergedGO',metric='TerminationStatus'",
                        checker_results,
                    )
                )
                go_converged = len(go_convergence_results) == 1 and go_convergence_results[0].success == "OK"
                if (
                    trajectory_result.success == "ENERGY_CHANGE"
                    and self.energy_change_frame == "next_if_go_converged"
                    and go_converged
                ):
                    stop += 1
                log_level(
                    "Keeping trajectory frames through {final_frame}",
                    level="DEBUG",
                    final_frame=stop - 1,
                )
                return ds[:stop]
        return ds


class RemoveInitialTraj(ConditionFilter):
    type: Literal["RemoveInitialTraj"] = "RemoveInitialTraj"
    skip_first_n_if_m_longer: Optional[Tuple[int, int]] = (2, 10)

    def run(self, ds: ConcreteInterfaces, checker_results: List[EngineCheckResult], **kwargs) -> ConcreteInterfaces:
        if self.skip_first_n_if_m_longer is not None and len(ds) >= self.skip_first_n_if_m_longer[1]:
            log_level(
                "Skip n if m longer {remove_initial}",
                level="DEBUG",
                remove_initial=self.skip_first_n_if_m_longer[0],
            )
            return ds[self.skip_first_n_if_m_longer[0] :]
        return ds


class DataSelector(ConditionFilter):
    type: Literal["DataSelector"] = "DataSelector"
    low_data_threshold: int = 100
    low_data_selector: Optional[Union[UnionFilters, SliceTuple]] = LinearSteppedFilter(step=-4, num_samples=2)
    default_data_selector: Union[UnionFilters, SliceTuple] = LinearSteppedFilter(step=-4, num_samples=3)

    def run(self, ds: ConcreteInterfaces, checker_results: List[EngineCheckResult], **kwargs) -> ConcreteInterfaces:
        if self.low_data_selector is not None and len(ds) <= self.low_data_threshold:
            log_level(
                "Running low data selection: found {n_data} <= {tr}",
                level="DEBUG",
                n_data=len(ds),
                tr=self.low_data_threshold,
            )
            return self.run_data_selector(ds, self.low_data_selector)
        else:
            return self.run_data_selector(ds, self.default_data_selector)

    def run_data_selector(self, ds, data_selector):
        dataset = ds[slice(*data_selector)] if isinstance(data_selector, tuple) else data_selector(ds)
        return dataset


class FractionalTrajectorySelector(ConditionFilter):
    """Select trajectory frames at fractions of its zero-based extent."""

    type: Literal["FractionalTrajectorySelector"] = "FractionalTrajectorySelector"
    fractions: Tuple[float, ...] = (1 / 3, 2 / 3, 1.0)
    min_unique_frames: int = 2

    def run(self, ds: ConcreteInterfaces, checker_results: List[EngineCheckResult], **kwargs) -> ConcreteInterfaces:
        if len(ds) == 0:
            return ds
        if any(fraction < 0.0 or fraction > 1.0 for fraction in self.fractions):
            raise ValueError(f"Trajectory fractions must be in [0, 1], found {self.fractions}")
        final_index = len(ds) - 1
        indices = np.unique(np.rint(np.asarray(self.fractions) * final_index).astype(np.int64))
        if indices.size < self.min_unique_frames:
            raise ValueError(
                "Fractional trajectory fallback requires at least "
                f"{self.min_unique_frames} distinct frames; found {indices.size}."
            )
        return ds.subset(indices=indices)


ConcreteConditionalFilter = Annotated[
    Union[RemoveUnreasonableTraj, RemoveInitialTraj, DataSelector, FractionalTrajectorySelector],
    Field(discriminator="type"),
]


class AMSTrajGetter(Getter[AMSTaskResult]):
    type: Literal["AMSTrajGetter"] = "AMSTrajGetter"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (AMSMDTask, AMSTask)
    condition_filters: List[ConcreteConditionalFilter] = [RemoveUnreasonableTraj(), RemoveInitialTraj(), DataSelector()]
    getter_id: str = "AMSTrajGetter"

    RemoveUnreasonableTraj: ClassVar[Type[RemoveUnreasonableTraj]] = RemoveUnreasonableTraj
    RemoveInitialTraj: ClassVar[Type[RemoveInitialTraj]] = RemoveInitialTraj
    DataSelector: ClassVar[Type[DataSelector]] = DataSelector
    FractionalTrajectorySelector: ClassVar[Type[FractionalTrajectorySelector]] = FractionalTrajectorySelector

    def run(self, result: AMSTaskResult, checker_results: List[EngineCheckResult], **kwargs) -> FrameGetterResult:
        return self.run_from_rkf_path(
            result=result,
            checker_results=checker_results,
            rkf_path=result.plams_results.rkfpath(),
            **kwargs,
        )

    def run_from_rkf_path(
        self,
        result: TaskResult,
        checker_results: List[EngineCheckResult],
        rkf_path: Optional[Union[str, Path]],
        getter_id: Optional[str] = None,
        **kwargs,
    ) -> FrameGetterResult:
        result_getter_id = getter_id or self.getter_id
        if rkf_path is None:
            return FrameGetterResult(
                engine_id=result.from_engine.engine_id,
                task_id=result.from_task.task_id,
                system_id=result.from_task.system_id,
                checker_id=(checker_results[-1].checker_id if len(checker_results) > 0 else "-empty-"),
                getter_id=result_getter_id,
                dataset=InMemoryMolData(),
            )
        ds = load_dataset(rkf_path)
        for c in self.condition_filters:
            ds = c.run(ds, checker_results, **kwargs)

        log_level("Selected: {data}", level="DEBUG", data=ds.absolute_idxs)
        return FrameGetterResult(
            engine_id=result.from_engine.engine_id,
            task_id=result.from_task.task_id,
            system_id=result.from_task.system_id,
            checker_id=(checker_results[-1].checker_id if len(checker_results) > 0 else "-empty-"),
            getter_id=result_getter_id,
            dataset=ds,
        )
