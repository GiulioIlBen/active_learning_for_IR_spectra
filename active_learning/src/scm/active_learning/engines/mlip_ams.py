from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional

from pydantic import JsonValue
from scm.moliterate import ConcreteInterfaces, PropertyInfo
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.filters import ConditionsFilter
from scm.moliterate.interfaces import ParAMSData

try:
    from scm.params import ParAMSResults as _ParAMSResults
except ModuleNotFoundError:
    _ParAMSResults = None
from scm.plams import AMSJob, Settings
from scm.plams.interfaces.adfsuite.ams import hybrid_committee_engine_settings

from scm.active_learning.task_parallelization import (
    ConcreteAMSParallelStrategies,
    ParallelStrategy,
)

from .ams import AMSEngine
from .core import Engine

if TYPE_CHECKING:
    from scm.params import ParAMSResults


class ParAMSEngine(Engine[ConcreteAMSParallelStrategies]):
    type: Literal["ParAMSEngine"] = "ParAMSEngine"
    source_settings: List[Dict[str, JsonValue]] = []
    finetunable: bool = True
    job_path: str
    nproc: Optional[int] = 1
    add_OMP_NUM_THREADS: Optional[int] = 1

    @classmethod
    def from_params_results(cls, params_results: "ParAMSResults", engine_id: str):
        if _ParAMSResults is None:
            raise ModuleNotFoundError(
                "scm.params is required for ParAMSEngine.from_params_results(), but it is not installed."
            )
        if params_results.job.path is None:
            raise ValueError(f"{params_results.job.path=} must be not None")
        ml_jobs = params_results.get_machine_learning_jobs()
        if len(ml_jobs) == 0:
            msg = params_results.job.get_errormsg()
            capture_error_out = (
                r"(?ms)^\[\d{2}\.\d{2}\|\d{2}:\d{2}:\d{2}\]\s+"
                r"(Traceback \(most recent call last\):.*?)"
                r"(?=^\[\d{2}\.\d{2}\|\d{2}:\d{2}:\d{2}\]\s+=+|\Z)"
            )
            msg2 = "\n".join(params_results.regex_file("$JN.out", capture_error_out))
            raise ValueError(
                f"ParAMSEngine.from_params_results found a failed ParAMSResults with:"
                f"\n\n----ErrorMsg\n{msg}\n\n----Output\n{msg2}"
            )
        return cls(
            engine_id=engine_id,
            source_settings=[
                x.results.get_production_engine_settings().as_dict() for x in params_results.get_machine_learning_jobs()
            ],
            finetunable=True,
            job_path=params_results.job.path,
        )

    def is_finetunable(self) -> bool:
        return self.finetunable

    @property
    def settings(self) -> Settings:
        if self.has_committee:
            # similar to implementation from scm.params.machine_learning.utils import get_committee_engine_settings
            s = hybrid_committee_engine_settings([x.settings for x in self.committee_members])
            if "Hybrid" in s.input and "engine forcefield" in AMSJob(settings=s).get_input().lower():
                s.input.Hybrid.GuessAttributesOnce = "No"
            return s
        e = Settings(self.source_settings[0])
        if self.nproc is not None:
            e.runscript.nproc = self.nproc
        if isinstance(self.add_OMP_NUM_THREADS, int):
            # similar to what hybrid_committee_engine_settings is doing
            if isinstance(e.runscript.preamble_lines, list):
                e.runscript.preamble_lines.append(f"export OMP_NUM_THREADS={self.add_OMP_NUM_THREADS}")
            else:
                e.runscript.preamble_lines = [f"export OMP_NUM_THREADS={self.add_OMP_NUM_THREADS}"]
        return e

    @property
    def has_committee(self) -> bool:
        # A single settings entry is not a committee; avoid recursive construction.
        return len(self.source_settings) > 1

    @property
    def committee_members(self) -> List["ParAMSEngine"]:
        if not self.has_committee:
            return []
        return [
            ParAMSEngine(
                engine_id=f"{self.engine_id[i]}",
                source_settings=[x],
                job_path=self.job_path,
            )
            for i, x in enumerate(self.source_settings)
        ]

    @property
    def ams_engine(self):
        return AMSEngine(
            engine_id=self.engine_id,
            source_settings=self.settings.as_dict(),
            finetunable=False,
        )

    def _validate_properties(self, requested_properties: List[PropertyInfo]) -> List[str]:
        return self.ams_engine._validate_properties(requested_properties)

    def run_single_point(
        self,
        properties: List[PropertyInfo],
        dataset: ConcreteInterfaces,
        parallel_settings: ParallelStrategy,
        **kwargs,
    ) -> Iterable[Dict[str | Literal["FAILURE"], Any | str]]:
        return self.ams_engine.run_single_point(
            properties=properties, dataset=dataset, parallel_settings=parallel_settings
        )

    @property
    def training_dataset(self) -> Optional[Iterable[ChemDataEntry]]:
        return self.get_ds_pop("training_set")

    @property
    def validation_dataset(self) -> Optional[Iterable[ChemDataEntry]]:
        return self.get_ds_pop("validation_set")

    @property
    def dataset(self):
        return ParAMSData(data_source=self.job_path)

    def get_ds_pop(self, name: str = "training_set"):
        """to reduce the memory size I prefer to pop the useless datasets"""
        ds = self.dataset
        dataset_selector = ConditionsFilter(
            conditions=f"{ParAMSData.dataset_metadata_key}={name}",
            collect_from="metadata",
        )
        ret = dataset_selector(ds)
        return ret
