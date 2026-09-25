from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, List, Tuple, Type, TypeVar

from pydantic import BaseModel

from scm.active_learning.results import EngineCheckResult, TaskResult
from scm.active_learning.tasks import Task

if TYPE_CHECKING:
    from scm.active_learning.checker_getters.getters.core import Getter

T = TypeVar("T", bound=TaskResult)


class Checker(BaseModel, Generic[T]):
    # type # Field discriminator
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = ...
    checker_id: str

    @abstractmethod
    def run(self, result: T, **kwargs) -> List[EngineCheckResult]: ...

    def collect_ids(self, result: T):
        return dict(
            task_id=result.from_task.task_id,
            system_id=result.from_task.system_id,
            engine_id=result.from_engine.engine_id,
            checker_id=self.checker_id,
        )

    def __add__(self, other: "Getter"):
        from scm.active_learning.checker_getters.decoupled_check_getter import (
            DecoupledCheckerGetter,
        )

        return DecoupledCheckerGetter(checker=self, getter=other)  # pyright: ignore[reportArgumentType]
