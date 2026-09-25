from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

from pydantic import BaseModel

from scm.active_learning.results import CheckGetResults, EngineCheckResult, TaskResult


class CheckerGetter(BaseModel):
    # type # Field discriminator
    # supported_tasks:ClassVar[List[Type[Task]]] = []

    @property
    def checker_id(self) -> str:
        return ""

    @property
    def getter_id(self) -> str:
        return ""

    @abstractmethod
    def run(
        self,
        result: TaskResult,
        c_res: Optional[List[EngineCheckResult]] = None,
        **kwargs,
    ) -> CheckGetResults: ...
