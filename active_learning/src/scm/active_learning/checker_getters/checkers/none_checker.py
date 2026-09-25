from __future__ import annotations

from typing import ClassVar, List, Literal, Tuple, Type

from scm.active_learning.checker_getters.checkers.core import Checker
from scm.active_learning.results import EngineCheckResult, TaskResult
from scm.active_learning.tasks import Task


class NoneChecker(Checker[TaskResult]):
    type: Literal["NoneChecker"] = "NoneChecker"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = ()
    checker_id: str = "NoneChecker"

    def run(self, result: TaskResult, **kwargs) -> List[EngineCheckResult]:
        return [
            EngineCheckResult(
                **self.collect_ids(result=result),  # pyright: ignore[reportArgumentType]
                property="None",
                metric="None",
                value=0,
                target=0,
                n_entries=0,
                success="OK",
            )
        ]
