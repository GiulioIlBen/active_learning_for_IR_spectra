from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List, Literal, Tuple, Type

from scm.moliterate import load_dataset
from scm.moliterate.interfaces import InMemoryMolData

from scm.active_learning.checker_getters.getters.ams_traj_getter import (
    ConcreteConditionalFilter,
    DataSelector,
    RemoveInitialTraj,
    RemoveUnreasonableTraj,
)
from scm.active_learning.checker_getters.getters.core import Getter
from scm.active_learning.logging import log_level
from scm.active_learning.results import EngineCheckResult, FrameGetterResult
from scm.active_learning.results.task.ase_md import ASEMolecularDynamicsResult
from scm.active_learning.tasks import ASEMolecularDynamics, Task


class ASETrajGetter(Getter[ASEMolecularDynamicsResult]):
    type: Literal["ASETrajGetter"] = "ASETrajGetter"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (ASEMolecularDynamics,)
    condition_filters: List[ConcreteConditionalFilter] = [
        RemoveUnreasonableTraj(),
        RemoveInitialTraj(),
        DataSelector(),
    ]
    getter_id: str = "ASETrajGetter"

    RemoveUnreasonableTraj: ClassVar[Type[RemoveUnreasonableTraj]] = RemoveUnreasonableTraj
    RemoveInitialTraj: ClassVar[Type[RemoveInitialTraj]] = RemoveInitialTraj
    DataSelector: ClassVar[Type[DataSelector]] = DataSelector

    def run(
        self,
        result: ASEMolecularDynamicsResult,
        checker_results: List[EngineCheckResult],
        **kwargs,
    ) -> FrameGetterResult:
        del kwargs
        trajectory_path = Path(result.trajectory_path)
        if not trajectory_path.is_file():
            return FrameGetterResult(
                engine_id=result.from_engine.engine_id,
                task_id=result.from_task.task_id,
                system_id=result.from_task.system_id,
                checker_id=(checker_results[-1].checker_id if len(checker_results) > 0 else "-empty-"),
                getter_id=self.getter_id,
                dataset=InMemoryMolData(),
            )
        ds = load_dataset(trajectory_path)
        for condition_filter in self.condition_filters:
            ds = condition_filter.run(ds, checker_results)

        log_level("Selected: {data}", level="DEBUG", data=ds.absolute_idxs)
        return FrameGetterResult(
            engine_id=result.from_engine.engine_id,
            task_id=result.from_task.task_id,
            system_id=result.from_task.system_id,
            checker_id=(checker_results[-1].checker_id if len(checker_results) > 0 else "-empty-"),
            getter_id=self.getter_id,
            dataset=ds,
        )
