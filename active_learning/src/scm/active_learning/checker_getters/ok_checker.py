from __future__ import annotations

from typing import List, Literal, Optional

from scm.active_learning.results import (
    CheckGetResults,
    EngineCheckResult,
    FrameGetterResult,
    TaskResult,
)

from .core import CheckerGetter


class OkCheckerGetter(CheckerGetter):
    type: Literal["OkCheckerGetter"] = "OkCheckerGetter"
    checker_id_: str = ""
    getter_id_: str = ""

    @property
    def checker_id(self) -> str:
        return self.checker_id_

    @property
    def getter_id(self) -> str:
        return self.getter_id_

    def run(
        self,
        result: TaskResult,
        c_res: Optional[List[EngineCheckResult]] = None,
        **kwargs,
    ) -> CheckGetResults:
        checks = c_res or []
        g_res = FrameGetterResult(
            engine_id=result.from_engine.engine_id,
            task_id=result.from_task.task_id,
            system_id=result.from_task.system_id,
            checker_id=self.checker_id,
            getter_id=self.getter_id,
        )
        return CheckGetResults(validity_results=checks, getter_results=g_res)
