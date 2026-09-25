from __future__ import annotations

import ast
from abc import abstractmethod
from typing import ClassVar, Generic, Iterable, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from scm.active_learning.results import EngineCheckResult, FrameGetterResult, TaskResult
from scm.active_learning.tasks import Task

T = TypeVar("T", bound=TaskResult)


class Getter(BaseModel, Generic[T]):
    # type # Field discriminator
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = ...
    getter_id: str

    @abstractmethod
    def run(self, result: T, checker_results: List[EngineCheckResult], **kwargs) -> FrameGetterResult: ...

    @classmethod
    def query_checker_results(
        cls, query: Optional[str], checker_results: List[EngineCheckResult]
    ) -> Iterable[EngineCheckResult]:
        if query is None:
            yield from checker_results
            return

        queries = query.split(",")
        couples = [q.split("=") for q in queries if "=" in q]
        couples_valid = [(k, ast.literal_eval(v)) for k, v in couples]

        def query_res(res):
            return all(getattr(res, c[0], None) == c[1] for c in couples_valid)

        for c in checker_results:
            if query_res(c):
                yield c
