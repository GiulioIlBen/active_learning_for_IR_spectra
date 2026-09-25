from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import JsonValue
from scm.plams import AMSJob, AMSMDJob, Settings

from scm.active_learning.engines import (
    AMSEngine,
    ASEEngine,
    ConcreteSystems,
    Engine,
    ParAMSEngine,
    SCMChemicalSystem,
    to_ams_engine,
)
from scm.active_learning.tasks.core import Task

if TYPE_CHECKING:
    from scm.active_learning.results.task.ams_task import AMSTaskResult

SupportedEnginesAMS = Union[AMSEngine, ParAMSEngine, ASEEngine]


class AMSTask(Task[SupportedEnginesAMS]):
    type: Literal["AMSTask"] = "AMSTask"  # discriminator
    task_id: str
    source_settings: Dict[str, JsonValue] = {}
    atomistic_system: ConcreteSystems

    @property
    def settings(self):
        return Settings(self.source_settings)

    @property
    def system_id(self) -> str:
        return self.atomistic_system.system_id

    def run(self, engine: Engine, *args, **amsjob_kwargs) -> "AMSTaskResult":
        from scm.active_learning.results.task.ams_task import AMSTaskResult

        engine = to_ams_engine(engine)
        # if not isinstance(engine, AMSEngine):
        # raise TypeError(f"{type(engine)=} is not valid, it should AMSEngine")
        j = self.build_job(engine=engine)
        res = j.run(**amsjob_kwargs)
        return AMSTaskResult(
            task=self,
            engine=engine,
            plams_results=res,
        )

    def build_job(self, engine: SupportedEnginesAMS, extra: Settings | None = None) -> AMSJob:
        """_summary_

        :param engine: _description_
        :type engine: AMSEngine
        :param extra: these are extra settings that are added without overriding pre-existing ones, defaults to None
        :type extra: Settings | None, optional
        :return: _description_
        :rtype: AMSJob
        """
        name = self.make_name(engine)

        extra = extra or Settings()
        return AMSJob(
            molecule=self.atomistic_system.collect_molecules(),
            settings=engine.settings + self.settings + extra,
            name=name,
        )

    def make_name(self, engine: SupportedEnginesAMS):
        for x, eng_id in zip(
            [engine.engine_id, self.system_id, self.task_id],
            ["engine_id", "system_id", "task_id"],
            strict=True,
        ):
            if "-" in x:
                raise ValueError(f"Ids should not contain `-` but found {x} in {eng_id} | {self.task_id=}")
        name = f"{engine.engine_id}-{self.system_id}-{self.task_id}"
        return name

    @classmethod
    def from_job(cls, job: AMSJob, task_id: str, system_id: str):
        if job.molecule is None and job.results is not None:
            job.molecule = job.results.get_input_molecule()
        assert job.molecule is not None
        return cls(
            task_id=task_id,
            source_settings=job.settings.as_dict(),
            atomistic_system=SCMChemicalSystem.from_molecules(system_id=system_id, systems=job.molecule),
        )

    def set_exit_conditions(
        self,
        exit_condition_freq: Optional[int] = None,
        minimum_distance: Optional[float] = 0.7,  # Ang
        engine_energy_uncertainty_per_atom_maybe: Optional[float] = 0.015,  # eV/NAtoms or eV
        engine_energy_uncertainty_normalization: Union[float, Literal["NAtoms"]] = "NAtoms",
        engine_gradients_uncertainty: Optional[float] = 0.5,  # eV/Ang
    ):
        """add in the source_settings extra specifications

        :param minimum_distance: in Angstrom, defaults to 0.7
        :type minimum_distance: Optional[float], optional
        :param engine_energy_uncertainty: in eV, defaults to 0.015
        :type engine_energy_uncertainty: Optional[float], optional
        :param engine_energy_normalization: _description_, defaults to "NAtoms"
        :type engine_energy_normalization: Union[float, Literal[&quot;NAtoms&quot;]], optional
        :param engine_gradients_uncertainty: in eV/Angstrom, defaults to 0.01580221, (i.e. 810 meV/Ang)
        :type engine_gradients_uncertainty: Optional[float], optional
        """
        # https://www.scm.com/doc/AMS/Tasks/Molecular_Dynamics.html#exit-conditions
        convert_eV_to_au = 0.019446903811488874

        s = Settings()
        driver_ams = s.input.ams
        if exit_condition_freq:
            driver_ams.MolecularDynamics.Trajectory.ExitConditionFreq = exit_condition_freq

        driver_ams.ExitCondition = []
        if minimum_distance is not None:
            driver_ams.ExitCondition.append(Settings())
            driver_ams.ExitCondition[-1].AtomsTooClose.MinimumDistance = minimum_distance
            driver_ams.ExitCondition[-1].Type = "AtomsTooClose"

        if engine_energy_uncertainty_per_atom_maybe is not None:
            driver_ams.ExitCondition.append(Settings())
            driver_ams.ExitCondition[-1].Type = "EngineEnergyUncertainty"
            driver_ams.ExitCondition[-1].EngineEnergyUncertainty.MaxUncertainty = (
                engine_energy_uncertainty_per_atom_maybe * convert_eV_to_au
            )
            if isinstance(engine_energy_uncertainty_normalization, float):
                driver_ams.ExitCondition[
                    -1
                ].EngineEnergyUncertainty.Normalization = engine_energy_uncertainty_normalization

        if engine_gradients_uncertainty is not None:
            driver_ams.ExitCondition.append(Settings())
            driver_ams.ExitCondition[-1].Type = "EngineGradientsUncertainty"
            driver_ams.ExitCondition[-1].EngineGradientsUncertainty.MaxUncertainty = (
                engine_gradients_uncertainty * convert_eV_to_au
            )

        s_new = self.settings
        s_new += s
        self.source_settings = s_new.as_dict()
        return self

    def set_seed(self, seed: Optional[int] = None):
        s = Settings()
        if seed:
            rng = np.random.default_rng(seed)
            s.input.ams.RNGSeed = " ".join([str(rng.integers(-999999999, 999999999)) for i in range(9)])
        self.source_settings = (self.settings + s).as_dict()
        return self

    @classmethod
    def from_amsmd_keywords(cls, **amsmd_kwargs) -> "AMSTask":
        md_task = AMSMDTask(**amsmd_kwargs)
        return cls(
            task_id=md_task.task_id,
            source_settings=md_task.settings.as_dict(),
            atomistic_system=md_task.atomistic_system,
        )

    @classmethod
    def from_amsgoir_keywords(cls, **amsgoir_kwargs) -> "AMSTask":
        goir_task = AMSGOIRTask(**amsgoir_kwargs)
        return cls(
            task_id=goir_task.task_id,
            source_settings=goir_task.settings.as_dict(),
            atomistic_system=goir_task.atomistic_system,
        )


class AMSMDTask(Task[SupportedEnginesAMS]):
    type: Literal["AMSMDTask"] = "AMSMDTask"  # discriminator
    task_id: str = "AMSMDTask"
    atomistic_system: ConcreteSystems

    nsteps: int = 1000
    timestep: float = 0.25
    samplingfreq: int = 100
    tau: Optional[int] = None  # get barostat_tau by multiplying the timestep with this number

    thermostat: Optional[Literal["NHC"]] = None
    temperature: Optional[Union[float, List[float]]] = None

    barostat: Optional[Literal["MTK"]] = None
    pressure: Optional[float] = None  # in MPa
    scale: Optional[Literal["XYZ"]] = None
    equal: Optional[str] = None
    constantvolume: Optional[bool] = None

    checkpointfrequency: Optional[int] = None
    writevelocities: Optional[bool] = True
    writebonds: Optional[bool] = True
    writecharges: Optional[bool] = True
    writemolecules: Optional[bool] = True
    writeenginegradients: Optional[bool] = None

    calcpressure: Optional[bool] = False
    binlog_time: Optional[bool] = False
    binlog_pressuretensor: Optional[bool] = False
    binlog_dipolemoment: Optional[bool] = False

    exit_condition_freq: Optional[int] = 1
    minimum_distance: Optional[float] = 0.7  # Ang
    engine_energy_uncertainty_per_atom_maybe: Optional[float] = None  # 0.015  # eV/NAtoms or eV
    engine_energy_uncertainty_normalization: Union[float, Literal["NAtoms"]] = "NAtoms"
    engine_gradients_uncertainty: Optional[float] = None  # 0.5  # eV/Ang

    @property
    def md_job(self):
        return AMSMDJob(
            **self.model_dump(
                exclude={
                    "type",
                    "task_id",
                    "atomistic_system",
                    "exit_condition_freq",
                    "minimum_distance",
                    "engine_energy_uncertainty_per_atom_maybe",
                    "engine_energy_uncertainty_normalization",
                    "engine_gradients_uncertainty",
                }
            )
        )

    @property
    def settings(self) -> Settings:
        return self.md_job.settings

    @property
    def system_id(self) -> str:
        return self.atomistic_system.system_id

    @property
    def ams_task(self):
        return AMSTask(
            task_id=self.task_id,
            source_settings=self.settings.as_dict(),
            atomistic_system=self.atomistic_system,
        ).set_exit_conditions(
            exit_condition_freq=self.exit_condition_freq,
            minimum_distance=self.minimum_distance,
            engine_energy_uncertainty_per_atom_maybe=self.engine_energy_uncertainty_per_atom_maybe,
            engine_energy_uncertainty_normalization=self.engine_energy_uncertainty_normalization,
            engine_gradients_uncertainty=self.engine_gradients_uncertainty,
        )

    def run(self, engine: Engine, *args, **amsjob_kwargs) -> "AMSTaskResult":
        return self.ams_task.run(engine, *args, **amsjob_kwargs)

    def build_job(self, engine: SupportedEnginesAMS, extra: Settings | None = None) -> AMSJob:
        return self.ams_task.build_job(engine, extra=extra)


class AMSGOIRTask(Task[SupportedEnginesAMS]):
    type: Literal["AMSGOIRTask"] = "AMSGOIRTask"  # discriminator
    task_id: str = "AMSGOIRTask"
    atomistic_system: ConcreteSystems

    task: Literal["GeometryOptimization"] = "GeometryOptimization"
    normal_modes: Optional[bool] = True
    dipole_moment: Optional[bool] = True
    max_restarts: Optional[int] = 5
    restart_displacement: Optional[float] = 0.05
    method: Optional[Literal["Quasi-Newton", "FIRE", "L-BFGS", "ConjugateGradients", "Dimer"]] = "Quasi-Newton"

    @property
    def settings(self) -> Settings:
        s = Settings()
        s.input.ams.Task = self.task
        if self.normal_modes is not None:
            s.input.ams.Properties.NormalModes = self.normal_modes
        if self.dipole_moment is not None:
            s.input.ams.Properties.DipoleMoment = self.dipole_moment
        if self.max_restarts is not None:
            s.input.ams.GeometryOptimization.MaxRestarts = self.max_restarts
        if self.restart_displacement is not None:
            s.input.ams.GeometryOptimization.RestartDisplacement = self.restart_displacement
        if self.method is not None:
            s.input.ams.GeometryOptimization.Method = self.method
        return s

    @property
    def system_id(self) -> str:
        return self.atomistic_system.system_id

    @property
    def ams_task(self):
        return AMSTask(
            task_id=self.task_id,
            source_settings=self.settings.as_dict(),
            atomistic_system=self.atomistic_system,
        )

    def run(self, engine: Engine, *args, **amsjob_kwargs) -> "AMSTaskResult":
        return self.ams_task.run(engine, *args, **amsjob_kwargs)

    def build_job(self, engine: SupportedEnginesAMS, extra: Settings | None = None) -> AMSJob:
        return self.ams_task.build_job(engine, extra=extra)
