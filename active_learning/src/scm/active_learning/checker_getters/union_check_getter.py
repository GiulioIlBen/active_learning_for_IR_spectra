from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel
from scm.moliterate import ConcatChemDataSet

from scm.active_learning.results import CheckGetResults, EngineCheckResult, FrameGetterResult, TaskResult

from .core import CheckerGetter
from .decoupled_check_getter import DecoupledCheckerGetter


class UnionMember(BaseModel):
    """A checker/getter pair inside a union, with the rule deciding whether its getter contributes frames."""

    check_getter: DecoupledCheckerGetter
    run_getter: Literal["always", "if_checks_pass", "never"] = "always"
    """``never`` makes the member check-only: its checks count, its getter contributes no frames."""

    @property
    def checker_id(self) -> str:
        return self.check_getter.checker_id

    @property
    def getter_id(self) -> str:
        return self.check_getter.getter_id

    def check(self, result: TaskResult) -> List[EngineCheckResult]:
        return self.check_getter.checker.run(result=result)

    def should_get(self, checks_so_far: List[EngineCheckResult]) -> bool:
        if self.run_getter == "always":
            return True
        if self.run_getter == "never":
            return False
        return all(check.is_success() for check in checks_so_far)

    def get(self, result: TaskResult, checks: List[EngineCheckResult]) -> FrameGetterResult:
        return self.check_getter.getter.run(result=result, checker_results=checks)


class UnionCheckerGetter(CheckerGetter):
    """Run every member checker and merge the frames of every member getter allowed to run.

    Unlike ``ConcatCheckerGetter`` (a cascade that keeps only one getter), the frames of all selected getters are
    returned together; each frame keeps the ``getter_id`` of the getter that produced it. A member with
    ``run_getter='if_checks_pass'`` contributes only when all checks collected up to and including its own passed.
    """

    type: Literal["UnionCheckerGetter"] = "UnionCheckerGetter"
    members: List[UnionMember]

    @property
    def checker_id(self) -> str:
        return "|".join(m.checker_id for m in self.members)

    @property
    def getter_id(self) -> str:
        return "|".join(m.getter_id for m in self.members)

    def run(
        self,
        result: TaskResult,
        c_res: Optional[List[EngineCheckResult]] = None,
        **kwargs,
    ) -> CheckGetResults:
        all_checks = list(c_res or [])
        getter_results: List[FrameGetterResult] = []
        for member in self.members:
            checks = member.check(result)
            all_checks.extend(checks)
            if member.should_get(all_checks):
                getter_results.append(member.get(result, checks))
        return CheckGetResults(
            validity_results=all_checks,
            getter_results=FrameGetterResult(
                engine_id=result.from_engine.engine_id,
                task_id=result.from_task.task_id,
                system_id=result.from_task.system_id,
                checker_id=self.checker_id,
                getter_id="|".join(g.getter_id for g in getter_results) or "-empty-",
                dataset=ConcatChemDataSet(data_source=[g.get_dataset_with_metadata() for g in getter_results]),
            ),
        )
