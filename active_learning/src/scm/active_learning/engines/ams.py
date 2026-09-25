from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import JsonValue

try:
    from scm.input_classes import drivers, engines
except ImportError:
    drivers = None
    engines = None
from scm.moliterate import ConcreteInterfaces, PropertyInfo
from scm.plams import AMSJob, Settings, jobs_in_directory

from scm.active_learning.engines.core import Engine
from scm.active_learning.task_parallelization import (
    ConcreteAMSParallelStrategies,
    ParallelStrategy,
)
from scm.active_learning.task_parallelization.ams_parallelization import (
    AMSParallelStrategy,
)


@dataclass
class AMSBobBuild:
    _value: AMSEngine

    def build(self):
        return self._value


class AMSEngineBuilder:
    @staticmethod
    def _require_input_classes():
        if drivers is None or engines is None:
            raise ImportError("scm.input_classes is required to use AMSEngineBuilder preset constructors.")
        return drivers, engines

    @staticmethod
    def B3LYP_TZP(engine_id=None, dispersion=True):
        input_drivers, input_engines = AMSEngineBuilder._require_input_classes()
        input_ams = input_drivers.AMS()
        input_ams.Engine = input_engines.ADF()
        input_ams.Engine.XC.Hybrid = "B3LYP"
        if dispersion:
            input_ams.Engine.XC.Dispersion = "GRIMME3 BJDAMP"
        input_ams.Engine.Basis.Type = "TZP"
        input_ams.Engine.Basis.Core = "None"
        s = Settings()
        s.input = input_ams.to_settings()
        return AMSBobBuild(
            AMSEngine(
                engine_id=engine_id or "B3LYP_TZP",
                source_settings=s.as_dict(),
            )
        )

    @staticmethod
    def XLYP_DZP(engine_id=None, higher_precision=False, dispersion=False):
        input_drivers, input_engines = AMSEngineBuilder._require_input_classes()
        input_ams = input_drivers.AMS()
        input_ams.Engine = input_engines.ADF()
        input_ams.Engine.XC.GGA = "XLYP"
        input_ams.Engine.Basis.Type = "DZP"
        if higher_precision:
            input_ams.Engine.AccurateGradients = True
            input_ams.Engine.NumericalQuality = "Good"
        if dispersion:
            input_ams.Engine.XC.Dispersion = "GRIMME3 BJDAMP"
        s = Settings()
        s.input = input_ams.to_settings()
        return AMSBobBuild(
            AMSEngine(
                engine_id=engine_id or "XLYP_DZP",
                source_settings=s.as_dict(),
            )
        )

    @staticmethod
    def LDA_DZP(engine_id=None):
        input_drivers, input_engines = AMSEngineBuilder._require_input_classes()
        input_ams = input_drivers.AMS()
        input_ams.Engine = input_engines.ADF()
        input_ams.Engine.XC.LDA = "LDA"
        input_ams.Engine.Basis.Type = "DZP"
        s = Settings()
        s.input = input_ams.to_settings()
        return AMSBobBuild(
            AMSEngine(
                engine_id=engine_id or "LDA_DZP",
                source_settings=s.as_dict(),
            )
        )

    @staticmethod
    def DFTB_GFN1(engine_id=None):
        input_drivers, input_engines = AMSEngineBuilder._require_input_classes()
        input_ams = input_drivers.AMS()
        input_ams.Engine = input_engines.DFTB()
        input_ams.Engine.Model = "GFN1-xTB"
        labelling_engine = Settings()
        labelling_engine.input = input_ams.to_settings()
        labelling_engine.runscript.nproc = 1
        s = Settings()
        s.input = input_ams.to_settings()
        return AMSBobBuild(
            AMSEngine(
                engine_id=engine_id or "DFTB_GFN1",
                source_settings=s.as_dict(),
            )
        )

    @staticmethod
    def UFF(engine_id=None, dipole_moment=True):
        input_drivers, input_engines = AMSEngineBuilder._require_input_classes()
        input_ams = input_drivers.AMS()
        input_ams.Engine = input_engines.ForceField()
        input_ams.Engine.Type = "UFF"
        input_ams.Engine.GuessCharges = dipole_moment
        s = Settings()
        s.input = input_ams.to_settings()
        return AMSBobBuild(
            AMSEngine(
                engine_id=engine_id or "UFF",
                source_settings=s.as_dict(),
            )
        )

    @staticmethod
    def GFNFF(engine_id=None):
        input_drivers, input_engines = AMSEngineBuilder._require_input_classes()
        input_ams = input_drivers.AMS()
        input_ams.Engine = input_engines.GFNFF()
        s = Settings()
        s.input = input_ams.to_settings()
        return AMSBobBuild(
            AMSEngine(
                engine_id=engine_id or "GFNFF",
                source_settings=s.as_dict(),
            )
        )

    @staticmethod
    def AIMNet2(
        engine_id=None,
        Model: Literal["AIMNet2-wB97MD3", "AIMNet2-B973c"] = "AIMNet2-wB97MD3",
        Device: Literal["cuda:0", "cpu"] = "cpu",
    ):
        input_drivers, input_engines = AMSEngineBuilder._require_input_classes()
        input_ams = input_drivers.AMS()
        input_ams.Engine = input_engines.MLPotential()
        input_ams.Engine.Model = Model
        input_ams.Engine.Device = Device
        s = Settings()
        s.input = input_ams.to_settings()
        return AMSBobBuild(
            AMSEngine(
                engine_id=engine_id or "AIMNet2",
                source_settings=s.as_dict(),
            )
        )


class AMSEngine(Engine[ConcreteAMSParallelStrategies]):
    type: Literal["AMSEngine"] = "AMSEngine"
    source_settings: Dict[str, JsonValue] = {}
    finetunable: bool = False
    duplicate_sp_policy: Literal["raise", "allow"] = "raise"

    def is_finetunable(self) -> bool:
        return self.finetunable

    @property
    def settings(self) -> Settings:
        return Settings(self.source_settings)

    def run_single_point(
        self,
        properties: List[PropertyInfo],
        dataset: ConcreteInterfaces,
        parallel_settings: ParallelStrategy,
        folder: Optional[str] = "SinglePoints",
        **kwargs,
    ) -> Iterable[Dict[str | Literal["FAILURE"], Any | str]]:
        # shared settings
        if isinstance(parallel_settings, AMSParallelStrategy):
            jr, s_nproc = parallel_settings.switch_parallel_plams()
        else:
            jr, s_nproc = None, Settings()
        s = s_nproc + self.settings
        s.input.ams.task = "SinglePoint"
        s += self._convert_property_to_ams_sp_settings(properties)

        # build jobs
        def build_job(row, i):
            j = AMSJob(
                molecule=row.chemical_system,
                settings=s,
                name=f"{self.engine_id}-{i}-SP",
            )
            return j

        jobs = [build_job(row, i) for i, row in enumerate(dataset)]
        if self.duplicate_sp_policy == "raise":
            first_index_by_hash: Dict[str, int] = {}
            duplicate_indices: Dict[int, int] = {}
            for index, job in enumerate(jobs):
                job_hash = job.hash_input()
                if job_hash in first_index_by_hash:
                    duplicate_indices[index] = first_index_by_hash[job_hash]
                else:
                    first_index_by_hash[job_hash] = index
            if duplicate_indices:
                duplicates = ", ".join(
                    f"{duplicate_index}->{original_index}"
                    for duplicate_index, original_index in duplicate_indices.items()
                )
                raise ValueError(
                    "Duplicate single-point inputs detected before AMS execution "
                    f"(duplicate->original indices: {duplicates})."
                )

        # run jobs
        job_dir_context = jobs_in_directory(path=folder) if folder else nullcontext()
        with job_dir_context:
            for j in jobs:
                j.run(jobrunner=jr)

        # yield results
        for j in jobs:
            yield self._extract_single_point_results(j, properties)

    ######################################################################################################
    #################################          Support          ##########################################
    ######################################################################################################

    def _convert_property_to_ams_sp_settings(self, properties: List[PropertyInfo]) -> Settings:
        valid_prop = self._validate_properties(properties)
        s = Settings()
        for p in valid_prop:
            if p != "Energy":
                s.input.ams.Properties[p] = True
        return s

    def _validate_properties(self, requested_properties: List[PropertyInfo]):
        names = set(x.name.lower() for x in requested_properties)
        mapping_names = self._from_property_to_ams_name
        extra_names = names.difference(set(str(x).lower() for x in mapping_names))
        if len(extra_names) > 0:
            raise ValueError(f"Not available properties: {extra_names} | Available are: {mapping_names.keys()}")
        return list(sorted(set([mapping_names[p.name] for p in requested_properties])))

    @property
    def _from_property_to_ams_name(self):
        """generic mapping, can be overridden for specific engine requirements"""
        return {
            "energy": "Energy",
            "Energy": "Energy",
            "Gradients": "Gradients",
            "Forces": "Gradients",
            "forces": "Gradients",
            "DipoleMoment": "DipoleMoment",
            "dipole_moment": "DipoleMoment",
            "dipole": "DipoleMoment",
            "DipoleGradients": "DipoleGradients",
            "atomic_polar_tensor": "DipoleGradients",
            "apt": "DipoleGradients",
            "dipole_gradients": "DipoleGradients",
            "dipole_derivatives": "DipoleGradients",
        }

    @property
    def _ams_propertie(self):
        return [
            PropertyInfo(name="Energy", unit="Hartree", shape="float"),
            PropertyInfo(name="Gradients", unit="Hartree/Bohr", shape=("nAtoms", 3)),
            PropertyInfo(name="DipoleMoment", unit="e*Bohr", shape=(3,)),
            PropertyInfo(name="DipoleGradients", unit="e", shape=("nAtoms", 3, 3)),
        ]

    def _extract_single_point_results(
        self,
        job: AMSJob,
        properties: List[PropertyInfo],
    ) -> Dict[str | Literal["FAILURE"], Any | str]:
        if not job.ok() or not job.check():
            return {"FAILURE": job.get_errormsg() or "Unknown error"}
        res: Dict[str | Literal["FAILURE"], Any | str] = {}
        gradients = None
        mapping_names = self._from_property_to_ams_name
        results = job.results

        for prop in properties:
            name = prop.name
            mapped = mapping_names[name]
            if name.lower() == "forces":
                if gradients is None:
                    gradients = results.get_gradients()
                res[name] = -gradients * prop.unit_conversion_from("Hartree/Bohr")
            elif mapped == "Gradients":
                if gradients is None:
                    gradients = results.get_gradients()
                res[name] = gradients * prop.unit_conversion_from("Hartree/Bohr")
            elif mapped == "Energy":
                res[name] = results.get_energy() * prop.unit_conversion_from("Hartree")
            elif mapped == "DipoleMoment":
                res[name] = results.get_dipolemoment() * prop.unit_conversion_from("e*Bohr")
            elif mapped == "DipoleGradients":
                dipole_gradients = results.get_dipolegradients().reshape(-1, 3, 3)
                res[name] = dipole_gradients * prop.unit_conversion_from("e")
        return res

    ######################################################################################################
    #################################         Builders          ##########################################
    ######################################################################################################

    @classmethod
    def from_amsjob(cls, job, engine_id: str):
        settings = job.settings.copy()
        if "ams" in settings.input:
            del settings.input["ams"]
        return cls.from_settings(
            settings=settings,
            engine_id=engine_id,
        )

    @classmethod
    def from_settings(cls, settings, engine_id: str):
        return cls(
            source_settings=settings.as_dict(),
            engine_id=engine_id,
        )

    class Builder(AMSEngineBuilder): ...
