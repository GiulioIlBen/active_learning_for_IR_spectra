from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional

from pydantic import JsonValue
from scm.plams import Settings

from scm.active_learning.engines import to_ams_engine
from scm.active_learning.engines.ams import AMSEngine
from scm.active_learning.tasks.core import Task

from ..engines.atomic_container import SCMChemicalSystem

if TYPE_CHECKING:
    from scm.conformers import ConformersJob

    from scm.active_learning.results.task.ams_conformers import AMSConformersResults


def _require_conformers_job():
    try:
        from scm.conformers import ConformersJob
    except ModuleNotFoundError as exc:
        if exc.name != "scm.conformers":
            raise
        raise ImportError("AMSConformersTask requires optional dependency `scm.conformers`.") from exc
    return ConformersJob


class AMSConformersTask(Task[AMSEngine]):
    type: Literal["AMSConformersTask"] = "AMSConformersTask"
    task_id: str
    source_settings: Dict[str, JsonValue] = {}
    atomistic_system: SCMChemicalSystem

    @property
    def settings(self) -> Settings:
        return Settings(self.source_settings)

    @property
    def system_id(self) -> str:
        return self.atomistic_system.system_id

    @classmethod
    def build_rdkit_settings(cls, InitialNConformers=3):
        s = Settings()
        driver = s.input.ams
        driver.Task = "Generate"
        driver.Generator.Method = "RDKit"
        driver.Generator.RDKit.InitialNConformers = InitialNConformers
        return s.as_dict()

    def run(self, engine: AMSEngine, *args, **conformers_kwargs) -> "AMSConformersResults":
        from scm.active_learning.results.task.ams_conformers import AMSConformersResults

        engine = to_ams_engine(engine)
        job = self.build_job(engine=engine)
        res = job.run(**conformers_kwargs)
        return AMSConformersResults(
            task=self,
            engine=engine,
            plams_results=res,
        )

    def build_job(self, engine: AMSEngine, extra: Optional[Settings] = None) -> ConformersJob:
        name = self.make_name(engine)
        extra = extra or Settings()
        conformers_job = _require_conformers_job()
        molecule = self.atomistic_system.collect_molecules()
        if len(molecule) > 1:
            raise ValueError("multi molecules not supported")
        return conformers_job(
            molecule=molecule[""],
            settings=engine.settings + self.settings + extra,
            name=name,
        )

    def make_name(self, engine: AMSEngine) -> str:
        for x, eng_id in zip(
            [engine.engine_id, self.system_id, self.task_id],
            ["engine_id", "system_id", "task_id"],
            strict=True,
        ):
            if "-" in x:
                raise ValueError(f"Ids should not contain `-` but found {x} in {eng_id} | {self.task_id=}")
        return f"{engine.engine_id}-{self.system_id}-{self.task_id}"

    @classmethod
    def from_job(cls, job: ConformersJob, task_id: str, system_id: str) -> "AMSConformersTask":
        _require_conformers_job()
        if job.molecule is None:
            raise ValueError("ConformersJob has no molecule")
        return cls(
            task_id=task_id,
            source_settings=job.settings.as_dict(),
            atomistic_system=SCMChemicalSystem.from_molecules(system_id=system_id, systems=job.molecule),
        )
