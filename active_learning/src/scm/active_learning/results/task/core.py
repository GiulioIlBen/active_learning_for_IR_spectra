from abc import abstractmethod

from pydantic import BaseModel

from scm.active_learning.engines.core import Engine
from scm.active_learning.tasks import Task


class TaskResult(BaseModel):
    # type: Literal[""] = ""
    @property
    @abstractmethod
    def from_task(self) -> Task: ...

    @property
    @abstractmethod
    def from_engine(self) -> Engine: ...
