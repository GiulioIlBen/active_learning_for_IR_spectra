from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple, Union

from pydantic import ConfigDict, field_serializer, field_validator
from scm.plams import Results, load

from scm.active_learning.engines import AMSEngine, ConcreteEngines
from scm.active_learning.tasks.ams_conformers_task import AMSConformersTask

from .core import TaskResult

try:
    from scm.conformers import ConformersResults

    _HAS_SCM_CONFORMERS = True
except ModuleNotFoundError as exc:
    if exc.name != "scm.conformers":
        raise
    ConformersResults = Results
    _HAS_SCM_CONFORMERS = False


def _require_scm_conformers() -> None:
    if not _HAS_SCM_CONFORMERS:
        raise ImportError("AMSConformersResults requires optional dependency `scm.conformers`.")


class AMSConformersResults(TaskResult):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["AMSConformersResults"] = "AMSConformersResults"
    task: AMSConformersTask
    engine: ConcreteEngines
    plams_results: ConformersResults

    @field_serializer("plams_results")
    def serialize_plams_results(self, plams_results: ConformersResults) -> str:
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
        plams_results: Union[Results, str],
        ids: Optional[Tuple[str, str, str]] = None,
    ) -> "AMSConformersResults":
        _require_scm_conformers()

        res = cls._get_results(plams_results)
        ids_coll = ids or cls._ids_from_job_name(res.job.name)
        return cls(
            task=AMSConformersTask.from_job(res.job, task_id=ids_coll[2], system_id=ids_coll[1]),
            engine=AMSEngine.from_amsjob(res.job, ids_coll[0]),
            plams_results=res,
        )

    @classmethod
    def _get_results(cls, plams_results: Union[Results, str]) -> ConformersResults:
        _require_scm_conformers()
        ret = plams_results
        if isinstance(plams_results, str):
            j = load(plams_results)
            if j is None:
                raise ValueError(f"{plams_results} can not be loaded")
            ret = j.results
        assert isinstance(ret, ConformersResults)
        return ret

    @staticmethod
    def _ids_from_job_name(name: str) -> Tuple[str, str, str]:
        parts = name.split("-")
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
        return "", "", ""

    @property
    def from_task(self):
        return self.task

    @property
    def from_engine(self):
        return self.engine
