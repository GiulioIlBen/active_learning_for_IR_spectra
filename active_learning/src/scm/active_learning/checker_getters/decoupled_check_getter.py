from __future__ import annotations

from typing import List, Literal, Optional

from scm.active_learning.results import CheckGetResults, EngineCheckResult, TaskResult

from .checkers import ConcreteChecker
from .core import CheckerGetter
from .getters import ConcreteGetter


class DecoupledCheckerGetter(CheckerGetter):
    type: Literal["DecoupledCheckerGetter"] = "DecoupledCheckerGetter"  # Field discriminator
    # supported_tasks:ClassVar[List[Type[Task]]] = []
    checker: ConcreteChecker
    getter: ConcreteGetter

    @property
    def getter_id(self) -> str:
        return self.getter.getter_id

    @property
    def checker_id(self) -> str:
        return self.checker.checker_id

    def run(
        self,
        result: TaskResult,
        c_res: Optional[List[EngineCheckResult]] = None,
        **kwargs,
    ) -> CheckGetResults:
        if c_res is None:
            c_res = self.checker.run(result=result)
        g_res = self.getter.run(result=result, checker_results=c_res)
        return CheckGetResults(validity_results=c_res, getter_results=g_res)

    def _not_pass(self, checker_results: List[EngineCheckResult]) -> bool:
        return not all(x.is_success() for x in checker_results)

    def __rshift__(self, other: "DecoupledCheckerGetter"):
        from scm.active_learning.checker_getters.concat_check_getter import (
            ConcatCheckerGetter,
        )

        return ConcatCheckerGetter(
            check_getters=[self, other],
        )
