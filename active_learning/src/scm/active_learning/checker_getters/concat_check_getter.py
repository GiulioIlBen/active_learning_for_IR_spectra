from __future__ import annotations

from typing import List, Literal, Optional

from scm.active_learning.results import CheckGetResults, EngineCheckResult, TaskResult

from .core import CheckerGetter
from .decoupled_check_getter import DecoupledCheckerGetter


class ConcatCheckerGetter(CheckerGetter):
    type: Literal["ConcatCheckerGetter"] = "ConcatCheckerGetter"
    check_getters: List[DecoupledCheckerGetter]

    @property
    def checker_id(self) -> str:
        return ">>".join(cg.checker_id for cg in self.check_getters)

    @property
    def getter_id(self) -> str:
        return ">>".join(cg.getter_id for cg in self.check_getters)

    def run(
        self,
        result: TaskResult,
        c_res: Optional[List[EngineCheckResult]] = None,
        **kwargs,
    ) -> CheckGetResults:
        all_checks = c_res or []
        for cg in self.check_getters:
            res = cg.checker.run(result=result)
            all_checks.extend(res)
            if cg._not_pass(res):
                ret = cg.run(result=result, c_res=res)
                ret.validity_results = all_checks
                return ret
        g_res = self.check_getters[-1].run(result=result, c_res=all_checks).getter_results
        return CheckGetResults(validity_results=all_checks, getter_results=g_res)

    def __rshift__(self, other: DecoupledCheckerGetter):
        return self.model_copy(update={"check_getters": [*self.check_getters, other]})
