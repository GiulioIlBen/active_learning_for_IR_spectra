from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List, Literal, Tuple, Type

import numpy as np
from pydantic import BaseModel, ConfigDict
from scm.plams import KFFile

from scm.active_learning.checker_getters.checkers.core import Checker
from scm.active_learning.results import EngineCheckResult
from scm.active_learning.results.task import AMSTaskResult
from scm.active_learning.tasks import AMSGOIRTask, AMSTask, Task


class HarmonicIRSpectrum(BaseModel):
    """Harmonic frequencies (cm^-1) and IR intensities (km/mol) of one engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: str
    frequencies: np.ndarray
    intensities: np.ndarray

    @classmethod
    def from_ams_results(cls, results, engine: str) -> "HarmonicIRSpectrum":
        return cls(
            engine=engine,
            frequencies=np.asarray(results.get_frequencies(engine=engine), dtype=float).ravel(),
            intensities=np.asarray(results.get_ir_intensities(engine=engine), dtype=float).ravel(),
        )

    def imaginary_indices(self, threshold: float) -> np.ndarray:
        return np.where(self.frequencies < threshold)[0]

    def broadened(self, grid: np.ndarray, width: float) -> np.ndarray:
        """Lorentzian-broadened spectrum on ``grid`` (paper Eq. 3); imaginary modes are left out."""
        real = self.frequencies > 0.0
        freqs = self.frequencies[real][:, None]
        ints = self.intensities[real][:, None]
        return (ints * width / np.pi / ((grid[None, :] - freqs) ** 2 + width**2)).sum(axis=0)

    def cos_ir(self, other: "HarmonicIRSpectrum", grid: np.ndarray, width: float) -> float:
        """Cosine similarity between the two broadened spectra (paper Eq. 4)."""
        a = self.broadened(grid, width)
        b = other.broadened(grid, width)
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b) / norm if norm > 0.0 else float("nan")


class AMSIRCommitteeAgreementChecker(Checker[AMSTaskResult]):
    """Paper validity checks for a GO-task run with a committee (Hybrid) engine.

    * Committee agreement: mean CosIR between each member spectrum and the spectrum of the committee-averaged Hessian
      (the Hybrid main engine) must be at least ``min_committee_agreement``.
    * No imaginary frequencies below ``imaginary_threshold`` beyond ``max_imaginary`` in the committee spectrum.
    """

    type: Literal["AMSIRCommitteeAgreementChecker"] = "AMSIRCommitteeAgreementChecker"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (AMSTask, AMSGOIRTask)
    checker_id: str = "IRCommittee"

    min_committee_agreement: float = 0.9
    broadening_width: float = 30.0
    grid_min: float = 0.0
    grid_max: float = 4000.0
    grid_step: float = 1.0
    imaginary_threshold: float = -10.0
    max_imaginary: int = 0

    @property
    def grid(self) -> np.ndarray:
        return np.arange(self.grid_min, self.grid_max + self.grid_step, self.grid_step)

    def run(self, result: AMSTaskResult, **kwargs) -> List[EngineCheckResult]:
        try:
            committee, members = self.load_spectra(result.plams_results)
        except Exception as exc:  # e.g. a GO that did not converge has no normal modes
            return [self._missing_spectra(result, exc)]
        return [self._check_imaginary(result, committee), self._check_agreement(result, committee, members)]

    @staticmethod
    def load_spectra(results) -> Tuple[HarmonicIRSpectrum, List[HarmonicIRSpectrum]]:
        results.collect_rkfs()
        main_engine = results.get_main_engine_name()
        AMSIRCommitteeAgreementChecker._collect_committee_member_rkfs(results, main_engine)
        committee = HarmonicIRSpectrum.from_ams_results(results, main_engine)
        members = [
            HarmonicIRSpectrum.from_ams_results(results, engine)
            for engine in results.engine_names()
            if engine != main_engine
        ]
        return committee, members

    @staticmethod
    def _collect_committee_member_rkfs(results, main_engine: str) -> None:
        """Register Hybrid member RKFs, which AMS does not list in ``ams.rkf``."""
        job_path = Path(results.job.path)
        for rkf_path in sorted(job_path.glob(f"{main_engine}-term*.rkf")):
            results.rkfs.setdefault(rkf_path.stem, KFFile(str(rkf_path)))

    def committee_agreement(self, committee: HarmonicIRSpectrum, members: List[HarmonicIRSpectrum]) -> float:
        grid = self.grid
        scores = [member.cos_ir(committee, grid, self.broadening_width) for member in members]
        return float(np.mean(scores)) if scores else float("nan")

    def _check_agreement(
        self, result: AMSTaskResult, committee: HarmonicIRSpectrum, members: List[HarmonicIRSpectrum]
    ) -> EngineCheckResult:
        value = self.committee_agreement(committee, members)
        success = bool(members) and value >= self.min_committee_agreement
        return EngineCheckResult(
            **self.collect_ids(result=result),  # pyright: ignore[reportArgumentType]
            property="IRSpect",
            metric="committee_CosIR",
            value=value,
            target=self.min_committee_agreement,
            units="d.u.",
            n_entries=len(members),
            lower_is_better=False,
            success="OK" if success else ("IRSpect_no_members" if not members else "IRSpect_committee_CosIR_failed"),
        )

    def _missing_spectra(self, result: AMSTaskResult, exc: Exception) -> EngineCheckResult:
        return EngineCheckResult(
            **self.collect_ids(result=result),  # pyright: ignore[reportArgumentType]
            property="IRSpect",
            metric="committee_CosIR",
            value=f"missing: {type(exc).__name__}",
            target=self.min_committee_agreement,
            units="d.u.",
            n_entries=0,
            lower_is_better=False,
            success="IRSpect_missing",
        )

    def _check_imaginary(self, result: AMSTaskResult, committee: HarmonicIRSpectrum) -> EngineCheckResult:
        n_imaginary = int(len(committee.imaginary_indices(self.imaginary_threshold)))
        return EngineCheckResult(
            **self.collect_ids(result=result),  # pyright: ignore[reportArgumentType]
            property="Freq",
            metric="n_imaginary",
            value=float(n_imaginary),
            target=float(self.max_imaginary),
            units="count",
            n_entries=len(committee.frequencies),
            lower_is_better=True,
            success="OK" if n_imaginary <= self.max_imaginary else "Freq_n_imaginary_failed",
        )
