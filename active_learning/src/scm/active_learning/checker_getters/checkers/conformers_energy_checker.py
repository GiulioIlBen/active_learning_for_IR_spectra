from typing import ClassVar, List, Literal, Optional, Tuple, Type

import numpy as np
import scm.plams as plams
from matplotlib import pyplot as plt
from pydantic import Field, PrivateAttr
from scm.moliterate import load_dataset

from scm.active_learning.checker_getters.checkers.core import Checker
from scm.active_learning.results import EngineCheckResult
from scm.active_learning.results.task import AMSConformersResults
from scm.active_learning.tasks import AMSConformersTask, Task

try:
    from scm.conformers import ConformersJob
except ModuleNotFoundError as exc:
    if exc.name != "scm.conformers":
        raise
    ConformersJob = None


def _require_conformers_job():
    if ConformersJob is None:
        raise ImportError("ConformersEnergyChecker requires optional dependency `scm.conformers`.")
    return ConformersJob


class ConfJobFailChecker(Checker[AMSConformersResults]):
    type: Literal["ConfJobFailChecker"] = "ConfJobFailChecker"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (AMSConformersTask,)
    checker_id: str = "ConfJFail"
    error_string: str = "Geometry optimization failed! (Did not converge.)"
    additional_error_strings: List[str] = [
        "Major bond changes after geometry optimization of input conformer.",
    ]
    ignored_error_strings: List[str] = []
    collect_geometry_errors: bool = True
    collect_worker_pool_cleanup_error: bool = False

    def run(self, result: AMSConformersResults, **kwargs) -> List[EngineCheckResult]:
        jobresults = result.plams_results
        error_msg = self._collect_error_message(jobresults)
        geometry_error = next(
            (
                message
                for message in [self.error_string, *self.additional_error_strings]
                if self.collect_geometry_errors and message in error_msg
            ),
            None,
        )
        worker_pool_cleanup_error = self.collect_worker_pool_cleanup_error and self._is_worker_pool_cleanup_error(
            error_msg
        )
        has_handled_error = geometry_error is not None or worker_pool_cleanup_error
        if error_msg and not has_handled_error and not self._is_ignored_error(error_msg):
            raise RuntimeError(
                f"ConfJobFailChecker: Unhandled conformer job error for {jobresults.job.name!r}: {error_msg}"
            )

        if not has_handled_error:
            try:
                load_dataset(jobresults.rkfpath())
            except Exception as exc:
                raise RuntimeError(
                    f"ConfJobFailChecker: Unhandled conformer results failure for {jobresults.job.name!r}: {exc}"
                ) from exc

        return [
            EngineCheckResult(
                **self.collect_ids(result=result),  # pyright: ignore[reportArgumentType]
                property="ConvergedGO",
                metric="ErrorMsg",
                value=error_msg,
                target="GOConvergence",
                units="bool",
                n_entries=1,
                value_type="str",
                success="OK" if not has_handled_error else ("GONoConv" if geometry_error else "AWPCleanupError"),
            )
        ]

    def _collect_error_message(self, jobresults) -> str:
        messages = []
        try:
            file_error = (jobresults.read_file("$JN.err") or "").strip()
        except Exception as exc:
            raise RuntimeError(
                f"Could not read '$JN.err' through PLAMS results for {jobresults.job.name!r}: {exc}"
            ) from exc
        if file_error and (
            self.collect_worker_pool_cleanup_error or not self._is_worker_pool_cleanup_error(file_error)
        ):
            messages.append(file_error)

        if self.collect_geometry_errors:
            for handled_error in [self.error_string, *self.additional_error_strings]:
                try:
                    output_matches = jobresults.grep_output(handled_error, options="-F")
                except Exception as exc:
                    raise RuntimeError(
                        "Could not search the conformer output through PLAMS results for "
                        f"{jobresults.job.name!r}: {exc}"
                    ) from exc
                messages.extend(line.strip() for line in output_matches if line.strip())

        plams_error = jobresults.job.get_errormsg() or ""
        if plams_error and (
            self.collect_worker_pool_cleanup_error or not self._is_worker_pool_cleanup_error(plams_error)
        ):
            messages.append(plams_error.strip())

        return "\n".join(dict.fromkeys(messages))

    @staticmethod
    def _is_worker_pool_cleanup_error(error_msg: str) -> bool:
        lines = [line.strip() for line in error_msg.strip().splitlines() if line.strip()]
        if not lines:
            return False
        return (
            lines[0] == "Traceback (most recent call last):"
            and any("weakref.py" in line and "_exitfunc" in line for line in lines)
            and any("shutil.py" in line and "rmtree" in line for line in lines)
            and lines[-1].startswith("FileNotFoundError:")
            and "/awp_" in lines[-1]
        )

    def _is_ignored_error(self, error_msg: str) -> bool:
        lines = [line.strip() for line in error_msg.splitlines() if line.strip()]
        return (
            bool(lines)
            and bool(self.ignored_error_strings)
            and all(any(ignored_error in line for ignored_error in self.ignored_error_strings) for line in lines)
        )


class ConformersEnergyChecker(Checker[AMSConformersResults]):
    type: Literal["ConformersEnergyChecker"] = "ConformersEnergyChecker"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (AMSConformersTask,)
    checker_id: str = "ConformersEnergy"
    energy_max_mean_std: Optional[float] = 0.02
    energy_max_std: Optional[float] = 0.03
    n_max_outliers: int = 0

    _std_energy: Optional[np.ndarray] = PrivateAttr(default=None)

    def run(
        self,
        result: AMSConformersResults,
        jobmanager=None,
        jobrunner=None,
        **kwargs,
    ) -> List[EngineCheckResult]:
        if self.energy_max_std is None and self.energy_max_mean_std is None:
            return []

        multijob = self.get_energy_score_multijob(result)
        multijob.run(jobmanager=jobmanager, jobrunner=jobrunner)
        energies = self.get_energies(multijob)
        std_energy = energies.std(axis=0)
        self._std_energy = std_energy

        n_entries = int(std_energy.size)
        metrics: List[EngineCheckResult] = []
        if self.energy_max_std is not None:
            n_out = int(np.count_nonzero(std_energy >= self.energy_max_std))
            metrics.append(
                self._make_result(
                    result=result,
                    prop="energy",
                    metric="n_max_SD",
                    value=float(n_out),
                    target=float(self.n_max_outliers),
                    units="eV",
                    n_entries=n_entries,
                    lower_is_better=True,
                    success=n_out <= self.n_max_outliers,
                )
            )

        if self.energy_max_mean_std is not None:
            mean_std_energy = float(std_energy.mean()) if n_entries > 0 else float("nan")
            metrics.append(
                self._make_result(
                    result=result,
                    prop="energy",
                    metric="mean_SD",
                    value=mean_std_energy,
                    target=float(self.energy_max_mean_std),
                    units="eV",
                    n_entries=n_entries,
                    lower_is_better=True,
                    success=mean_std_energy <= self.energy_max_mean_std,
                )
            )
        return metrics

    def get_energy_score_multijob(self, result: AMSConformersResults):
        jobresults = result.plams_results
        engines = result.engine.committee_members
        conformers_job = _require_conformers_job()
        score_jobs = []
        for i, engine_i in enumerate(engines):
            score_job = conformers_job(name=f"score_Eng{i}")
            score_job.settings.input.ams.Task = "Score"
            score_job.settings.input.ams.InputConformersSet = jobresults.rkfpath()
            score_job.settings += engine_i.settings
            score_jobs.append(score_job)
        multijob = plams.MultiJob(children=score_jobs, name="score_jobs")
        return multijob

    def get_energies(self, multijob):
        energies_score_all = []
        for c in multijob.children:
            energies_score = c.results.get_relative_energies(unit="ev")
            energies_score_all.append(energies_score)
        return np.asarray(energies_score_all)

    def plot_std(self):
        return plt.hist(self._std_energy)

    def _make_result(
        self,
        result: AMSConformersResults,
        prop: str,
        metric: str,
        value: float,
        target: float,
        units: str,
        n_entries: int,
        lower_is_better: bool,
        success: bool,
    ) -> EngineCheckResult:
        return EngineCheckResult(
            **self.collect_ids(result=result),  # pyright: ignore[reportArgumentType]
            property=prop,
            metric=metric,
            value=value,
            target=target,
            units=units,
            n_entries=n_entries,
            lower_is_better=lower_is_better,
            success="OK" if success else f"{prop}_{metric}_failed",
        )


class ConformersChecker(Checker[AMSConformersResults]):
    type: Literal["ConformersChecker"] = "ConformersChecker"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (AMSConformersTask,)
    checker_id: str = "ConformersChecker"
    energy_checker: ConformersEnergyChecker = Field(default_factory=ConformersEnergyChecker)
    fail_checker: ConfJobFailChecker = Field(default_factory=ConfJobFailChecker)
    skip_energy_on_failure: bool = True

    def run(
        self,
        result: AMSConformersResults,
        jobmanager=None,
        jobrunner=None,
        **kwargs,
    ) -> List[EngineCheckResult]:
        fail_results = self.fail_checker.run(result=result)
        if self.skip_energy_on_failure and any(r.success != "OK" for r in fail_results):
            return fail_results
        energy_results = self.energy_checker.run(
            result=result,
            jobmanager=jobmanager,
            jobrunner=jobrunner,
            **kwargs,
        )
        return fail_results + energy_results
