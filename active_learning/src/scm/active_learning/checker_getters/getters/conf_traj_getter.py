from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List, Literal, Tuple, Type

from pydantic import Field

from scm.active_learning.checker_getters.getters.ams_traj_getter import AMSTrajGetter
from scm.active_learning.checker_getters.getters.core import Getter
from scm.active_learning.results import EngineCheckResult, FrameGetterResult
from scm.active_learning.results.task import AMSConformersResults
from scm.active_learning.tasks import AMSConformersTask, Task


class ConfTrajGetter(Getter[AMSConformersResults]):
    type: Literal["ConfTrajGetter"] = "ConfTrajGetter"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (AMSConformersTask,)
    getter_id: str = "ConfTrajGetter"
    job: AMSTrajGetter = Field(default_factory=AMSTrajGetter)

    def run(
        self,
        result: AMSConformersResults,
        checker_results: List[EngineCheckResult],
        **kwargs,
    ) -> FrameGetterResult:
        job_path = result.plams_results.job.path
        if job_path is None:
            raise RuntimeError("Cannot locate initGO/ams.rkf because the conformer job has no path.")

        rkf_path = Path(job_path) / "initGO" / "ams.rkf"
        if not rkf_path.is_file():
            raise FileNotFoundError(f"Expected conformer initialization trajectory does not exist: {rkf_path}")

        return self.job.run_from_rkf_path(
            result=result,
            checker_results=checker_results,
            rkf_path=rkf_path,
            getter_id=self.getter_id,
            **kwargs,
        )
