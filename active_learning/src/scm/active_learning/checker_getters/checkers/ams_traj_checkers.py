from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, FrozenSet, List, Literal, Optional, Tuple, Type

from scm.active_learning.checker_getters.checkers.core import Checker
from scm.active_learning.results import EngineCheckResult
from scm.active_learning.results.task.ams_task import AMSTaskResult
from scm.active_learning.tasks import AMSMDTask, AMSTask, Task


class TrajectoryIssueReason(str, Enum):
    BROKEN_BOND = "BROKEN_BOND"
    DELTA_TEMPERATURE = "DELTA_TEMPERATURE"
    HIGH_TEMPERATURE = "HIGH_TEMPERATURE"
    ENERGY_CHANGE = "ENERGY_CHANGE"
    ENERGY_UNCERTAINTY = "ENERGY_UNCERTAINTY"
    GRADIENTS_UNCERTAINTY = "GRADIENTS_UNCERTAINTY"
    DISTANCE = "DISTANCE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class _TrajectoryIssue:
    frame: int
    reason: TrajectoryIssueReason


class AMSTrajChecker(Checker[AMSTaskResult]):
    type: Literal["AMSTrajChecker"] = "AMSTrajChecker"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (AMSMDTask, AMSTask)
    checker_id: str = "AMSTrajChecker"
    check_bond_breaking: bool = True
    max_delta_energy_per_atom_hartree: Optional[float] = 0.005
    max_temperature: Optional[float] = 5000.0
    max_delta_temperature: Optional[float] = None
    check_geometry_optimization_convergence: bool = False

    def run(self, result: AMSTaskResult, **kwargs) -> List[EngineCheckResult]:
        final_frame = self._get_expected_final_frame(result)
        n_entries = self._n_entries(result)
        issue = self._find_first_issue(result.plams_results, n_entries)
        reasonable_frame = issue.frame if issue is not None else final_frame
        success = "OK" if issue is None else issue.reason.value
        check_results = [
            EngineCheckResult(
                **self.collect_ids(result=result),  # pyright: ignore[reportArgumentType]
                property="trajectory",
                units="0-index",
                metric="reasonable_final_frame",
                value=reasonable_frame - 1,
                target=final_frame - 1,
                n_entries=n_entries,
                success=success,
            )
        ]
        if self.check_geometry_optimization_convergence:
            convergence_result = self._geometry_optimization_convergence_result(result)
            if convergence_result is not None:
                check_results.append(convergence_result)
        return check_results

    def _geometry_optimization_convergence_result(self, result: AMSTaskResult) -> Optional[EngineCheckResult]:
        job = result.plams_results.job
        task = job.get_task()
        if task is None or task.lower() != "geometryoptimization":
            return None
        is_converged = job.ok() and job.check()
        try:
            termination_status = str(result.plams_results.readrkf("General", "termination status"))
        except Exception:
            termination_status = job.get_errormsg() or "Unknown termination status"
        return EngineCheckResult(
            **self.collect_ids(result=result),  # pyright: ignore[reportArgumentType]
            property="ConvergedGO",
            units="bool",
            metric="TerminationStatus",
            value=termination_status,
            target="GOConvergence",
            n_entries=1,
            value_type="str",
            success="OK" if is_converged else "GONoConv",
        )

    def _find_first_issue(self, results, final_frame: int) -> Optional[_TrajectoryIssue]:
        issues: List[_TrajectoryIssue] = []
        if self.check_bond_breaking:
            issue = self._bond_break_issue(results, final_frame)
            if issue is not None:
                issues.append(issue)
        if self.max_temperature is not None or self.max_delta_temperature is not None:
            issue = self._temperature_issue(results)
            if issue is not None:
                issues.append(issue)
        if self.max_delta_energy_per_atom_hartree is not None:
            issue = self._energy_issue(results)
            if issue is not None:
                issues.append(issue)
        exit_issue = self._exit_condition_issue(results, final_frame)
        if exit_issue is not None:
            issues.append(exit_issue)
        if not issues:
            return None
        return min(issues, key=lambda x: x.frame)

    def _bond_break_issue(self, results, final_frame: int) -> Optional[_TrajectoryIssue]:
        if final_frame <= 1:
            return None
        initial = self._get_history_molecule(results, 1)
        initial_bonds = self._bond_pairs(initial)
        if not initial_bonds:
            return None
        for step in range(2, final_frame + 1):
            molecule = self._get_history_molecule(results, step)
            if molecule is None:
                continue
            if not initial_bonds.issubset(self._bond_pairs(molecule)):
                return _TrajectoryIssue(frame=max(1, step - 1), reason=TrajectoryIssueReason.BROKEN_BOND)
        return None

    def _temperature_issue(self, results) -> Optional[_TrajectoryIssue]:
        temperatures = self._get_history_property(results, "Temperature", history_section="MDHistory")
        if temperatures is None or len(temperatures) == 0:
            return None
        for idx, temp in enumerate(temperatures, start=1):
            if self.max_temperature is not None and temp > self.max_temperature:
                return _TrajectoryIssue(frame=max(1, idx - 1), reason=TrajectoryIssueReason.HIGH_TEMPERATURE)
            if (
                self.max_delta_temperature is not None
                and idx > 2
                and abs(temp - temperatures[idx - 2]) > self.max_delta_temperature
            ):
                return _TrajectoryIssue(
                    frame=max(1, idx - 1),
                    reason=TrajectoryIssueReason.DELTA_TEMPERATURE,
                )
        return None

    def _energy_issue(self, results) -> Optional[_TrajectoryIssue]:
        energies = self._get_history_property(results, "EngineEnergy")
        if energies is None:
            energies = self._get_history_property(results, "Energy")
        if energies is None or len(energies) == 0:
            return None
        n_atoms = self._get_atom_count(results)
        if n_atoms is None:
            return None
        limit = self.max_delta_energy_per_atom_hartree * n_atoms  # type: ignore[operator]
        baseline = energies[0]
        for idx, energy in enumerate(energies, start=1):
            if abs(energy - baseline) > limit:
                return _TrajectoryIssue(frame=max(1, idx - 1), reason=TrajectoryIssueReason.ENERGY_CHANGE)
        return None

    def _exit_condition_issue(self, results, final_frame: int) -> Optional[_TrajectoryIssue]:
        message = self._get_exit_condition_message(results)
        if not message:
            return None
        msg = message.lower()
        if "atoms too close" in msg:
            return _TrajectoryIssue(frame=max(final_frame - 1, 1), reason=TrajectoryIssueReason.DISTANCE)
        if "engine energy gradients uncertainty" in msg:
            return _TrajectoryIssue(frame=final_frame, reason=TrajectoryIssueReason.GRADIENTS_UNCERTAINTY)
        if "engine energy uncertainty" in msg:
            return _TrajectoryIssue(frame=final_frame, reason=TrajectoryIssueReason.ENERGY_UNCERTAINTY)
        return _TrajectoryIssue(frame=final_frame, reason=TrajectoryIssueReason.UNKNOWN)

    @staticmethod
    def _get_expected_final_frame(results: AMSTaskResult) -> int:
        previous = AMSTrajChecker._previous_task_results(results)
        NSteps_copied = AMSTrajChecker._n_steps_from_settings(previous) if previous else 0
        n_entries_from_settings = AMSTrajChecker._n_entries_from_settings(results, n_steps_copied=NSteps_copied or 0)
        if n_entries_from_settings:
            # add the initial frame back again
            extra = AMSTrajChecker._n_entries(previous) if previous else 0
            return n_entries_from_settings + extra
        return AMSTrajChecker._n_entries(results)

    @staticmethod
    def _n_entries(results: AMSTaskResult) -> int:
        try:
            ret = int(results.plams_results.readrkf("History", "nEntries"))
            return ret
        except Exception:
            return -1

    @staticmethod
    def _n_entries_from_settings(results: AMSTaskResult, n_steps_copied=0):
        plams_res = results.plams_results
        NSteps = AMSTrajChecker._n_steps_from_settings(results)
        sampling_freq = plams_res.job.settings.get_nested(
            tuple(["input", "ams", "moleculardynamics", "Trajectory", "SamplingFreq"]),
            default=None,
        )
        # I add one to NSteps, since the final frames in the rkf is the NSteps+initial frame = nEntries
        # So consistent with nEntries
        if NSteps is not None and sampling_freq is not None:
            n_frames_copied = int(n_steps_copied) // int(sampling_freq)
            if n_frames_copied > 0:
                n_frames_copied += 1
            return int(NSteps) // int(sampling_freq) - n_frames_copied + 1
        return None

    @staticmethod
    def _n_steps_from_settings(results: AMSTaskResult):
        NSteps = results.plams_results.job.settings.get_nested(
            tuple(["input", "ams", "moleculardynamics", "NSteps"]),
            default=None,
        )
        return NSteps

    @staticmethod
    def _previous_task_results(results: AMSTaskResult) -> Optional[AMSTaskResult]:
        plams_res = results.plams_results
        copy_restart_traj = plams_res.job.settings.get_nested(
            tuple(["input", "ams", "moleculardynamics", "CopyRestartTrajectory"]),
            default=None,
        )
        if copy_restart_traj in ["Yes", "yes", True]:
            restart_path = plams_res.job.settings.get_nested(
                tuple(["input", "ams", "moleculardynamics", "Restart"]),
                default=None,
            )
            if isinstance(restart_path, str):
                # you do not copy th initial frame
                dill_path = Path(restart_path).parent / (Path(restart_path).parent.name + ".dill")
                if dill_path.is_file():
                    try:
                        return AMSTaskResult.from_results(dill_path)
                    except Exception:
                        pass
        return None

    @staticmethod
    def _get_history_property(results, key: str, history_section: Optional[str] = None):
        try:
            if history_section is None:
                return results.get_history_property(key)
            return results.get_history_property(key, history_section=history_section)
        except KeyError:
            return None

    @staticmethod
    def _get_history_molecule(results, step: int):
        try:
            return results.get_history_molecule(step)
        except Exception:
            return None

    @staticmethod
    def _bond_pairs(molecule) -> FrozenSet[Tuple[int, int]]:
        if molecule is None:
            return frozenset()
        molecule.set_atoms_id()
        return frozenset(tuple(sorted((bond.atom1.id, bond.atom2.id))) for bond in molecule.bonds)

    @staticmethod
    def _get_exit_condition_message(results) -> str:
        try:
            return results.get_exit_condition_message()
        except Exception:
            return ""

    @staticmethod
    def _get_atom_count(results) -> Optional[int]:
        try:
            return len(results.get_main_molecule())
        except Exception:
            return None
