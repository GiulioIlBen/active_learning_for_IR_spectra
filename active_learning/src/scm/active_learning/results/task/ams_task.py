from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from pydantic import ConfigDict, field_serializer, field_validator
from scm.plams import AMSJob, AMSResults, Results, load

from scm.active_learning.engines import AMSEngine, ConcreteEngines
from scm.active_learning.tasks import AMSTask

from .core import TaskResult


class AMSTaskResult(TaskResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["AMSTaskResult"] = "AMSTaskResult"
    task: AMSTask
    engine: ConcreteEngines
    plams_results: AMSResults

    @field_serializer("plams_results")
    def serialize_plams_results(self, plams_results: AMSResults) -> str:
        if plams_results.job.path is None:
            return ""
        return plams_results.job.path

    @field_validator("plams_results", mode="before")
    @classmethod
    def validate_plams_results(cls, plams_results: Union[Results, str]) -> Results:
        if isinstance(plams_results, str):
            j = load(plams_results + f"/{Path(plams_results).name}.dill")
            if j is None:
                raise ValueError(f"{plams_results} can not be loaded")
            return j.results
        return plams_results

    @classmethod
    def from_results(
        cls,
        plams_results: Union[Results, str, Path],
        ids: Optional[Tuple[str, str, str]] = None,
    ) -> "AMSTaskResult":
        """_summary_

        :param plams_results: _description_
        :type plams_results: Union[Results, str]
        :param ids: in order: engine_id,system_id,task_id, defaults to None
        :type ids: Optional[Tuple[str,str,str]], optional
        :return: _description_
        :rtype: AMSTaskResult
        """
        res = cls._get_results(plams_results)
        ids_coll = ids or res.job.name.split("-")
        return cls(
            task=AMSTask.from_job(res.job, task_id=ids_coll[2], system_id=ids_coll[1]),
            engine=AMSEngine.from_amsjob(res.job, ids_coll[0]),
            plams_results=res,
        )

    @classmethod
    def _get_results(cls, plams_results: Union[Results, str, Path]) -> AMSResults:
        ret = plams_results
        if isinstance(plams_results, (str, Path)):
            j = load(str(plams_results))
            if j is None:
                raise ValueError(f"{plams_results} can not be loaded")
            ret = j.results
        assert isinstance(ret, AMSResults)
        return ret

    @property
    def from_task(self):
        return self.task

    @property
    def from_engine(self):
        return self.engine

    @property
    def executed_job(self) -> Optional[AMSJob]:
        return getattr(self.plams_results, "job", None)
