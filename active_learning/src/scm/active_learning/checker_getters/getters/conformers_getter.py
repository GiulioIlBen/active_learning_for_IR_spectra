from __future__ import annotations

from typing import ClassVar, List, Literal, Optional, Tuple, Type

from scm.moliterate import load_dataset
from scm.moliterate.filters import RandomFilter
from scm.moliterate.interfaces import InMemoryMolData

from scm.active_learning.checker_getters.getters.core import Getter
from scm.active_learning.results import EngineCheckResult, FrameGetterResult, TaskResult
from scm.active_learning.tasks import Task


class ConformersGetter(Getter[TaskResult]):
    type: Literal["ConformersGetter"] = "ConformersGetter"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (Task,)
    getter_id: str = "ConformersGetter"
    max_n_structures: int = -1
    seed: Optional[int] = None

    def run(self, result: TaskResult, checker_results: List[EngineCheckResult], **kwargs) -> FrameGetterResult:
        plams_results = getattr(result, "plams_results", None)
        rkf_path = plams_results.rkfpath() if plams_results is not None else None
        if rkf_path is None:
            dataset = InMemoryMolData()
        else:
            ds = load_dataset(rkf_path)
            if self.max_n_structures < 0:
                dataset = ds
            elif self.max_n_structures == 0:
                dataset = ds.subset(indices=[])
            else:
                dataset = RandomFilter(num_samples=self.max_n_structures, seed=self.seed)(ds)

        return FrameGetterResult(
            engine_id=result.from_engine.engine_id,
            task_id=result.from_task.task_id,
            system_id=result.from_task.system_id,
            checker_id=(checker_results[-1].checker_id if len(checker_results) > 0 else "-empty-"),
            getter_id=self.getter_id,
            dataset=dataset,
        )
