from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Tuple

from pydantic import BaseModel

if TYPE_CHECKING:
    from scm.active_learning.engines import ConcreteEngines
    from scm.active_learning.results import ConcreteTaskResult
    from scm.active_learning.tasks import ConcreteTask

# T = TypeVar("T", bound=Tuple[Task, Engine])
# R = TypeVar("R", bound=TaskResult)


class ParallelStrategy(BaseModel):
    # type: str  # It must have a type, and it is used as discriminator

    @abstractmethod
    def run_tasks(
        self, task_engine_coll: List[Tuple["ConcreteTask", "ConcreteEngines"]]
    ) -> List["ConcreteTaskResult"]: ...
