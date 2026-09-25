from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, FrozenSet, Literal, Optional, Tuple, Type

from scm.moliterate import load_dataset

from scm.active_learning.checker_getters.checkers.core import Checker
from scm.active_learning.results import EngineCheckResult
from scm.active_learning.results.task.ase_md import ASEMolecularDynamicsResult
from scm.active_learning.tasks import ASEMolecularDynamics, Task


class TrajectoryIssueReason(str, Enum):
    BROKEN_BOND = "BROKEN_BOND"
    DISTANCE = "DISTANCE"
    GRADIENTS_UNCERTAINTY = "GRADIENTS_UNCERTAINTY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class _TrajectoryIssue:
    frame: int
    reason: TrajectoryIssueReason


class ASETrajChecker(Checker[ASEMolecularDynamicsResult]):
    type: Literal["ASETrajChecker"] = "ASETrajChecker"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (ASEMolecularDynamics,)
    checker_id: str = "ASETrajChecker"
    check_bond_breaking: bool = True

    def run(self, result: ASEMolecularDynamicsResult, **kwargs) -> list[EngineCheckResult]:
        del kwargs
        dataset = self._load_dataset(result.trajectory_path)
        n_entries = len(dataset)
        if n_entries == 0:
            return [
                EngineCheckResult(
                    **self.collect_ids(result=result),
                    property="trajectory",
                    units="0-index",
                    metric="reasonable_final_frame",
                    value_type="str",
                    value="",
                    target="",
                    n_entries=0,
                    success="NO_DATA",
                )
            ]

        final_frame = n_entries
        issue = self._find_first_issue(result=result, final_frame=final_frame)
        reasonable_frame = issue.frame if issue is not None else final_frame
        success = "OK" if issue is None else issue.reason.value
        return [
            EngineCheckResult(
                **self.collect_ids(result=result),
                property="trajectory",
                units="0-index",
                metric="reasonable_final_frame",
                value=reasonable_frame - 1,
                target=final_frame - 1,
                n_entries=n_entries,
                success=success,
            )
        ]

    def _find_first_issue(self, result: ASEMolecularDynamicsResult, final_frame: int) -> Optional[_TrajectoryIssue]:
        issues: list[_TrajectoryIssue] = []
        if self.check_bond_breaking:
            issue = self._bond_break_issue(result.trajectory_path, final_frame)
            if issue is not None:
                issues.append(issue)
        stop_issue = self._stop_issue(result=result, final_frame=final_frame)
        if stop_issue is not None:
            issues.append(stop_issue)
        if not issues:
            return None
        return min(issues, key=lambda item: item.frame)

    def _bond_break_issue(self, trajectory_path: str, final_frame: int) -> Optional[_TrajectoryIssue]:
        if final_frame <= 1:
            return None
        dataset = self._load_dataset(trajectory_path)
        initial = dataset.get_row(0).molecule
        initial_bonds = self._bond_pairs(initial)
        if not initial_bonds:
            return None
        for idx in range(1, final_frame):
            molecule = dataset.get_row(idx).molecule
            if not initial_bonds.issubset(self._bond_pairs(molecule)):
                return _TrajectoryIssue(frame=max(1, idx), reason=TrajectoryIssueReason.BROKEN_BOND)
        return None

    @staticmethod
    def _stop_issue(result: ASEMolecularDynamicsResult, final_frame: int) -> Optional[_TrajectoryIssue]:
        if not result.stop_triggered or len(result.stop_reasons) == 0:
            return None
        if final_frame <= 0:
            return None

        raw_reason = result.stop_reasons[0]
        if raw_reason.startswith("minimum_distance<"):
            reason = TrajectoryIssueReason.DISTANCE
        elif raw_reason.startswith("relative_force_error>"):
            reason = TrajectoryIssueReason.GRADIENTS_UNCERTAINTY
        else:
            reason = TrajectoryIssueReason.UNKNOWN
        return _TrajectoryIssue(frame=max(1, final_frame - 1), reason=reason)

    @staticmethod
    def _load_dataset(trajectory_path: str):
        return load_dataset(Path(trajectory_path))

    @staticmethod
    def _bond_pairs(molecule) -> FrozenSet[Tuple[int, int]]:
        if molecule is None:
            return frozenset()
        if hasattr(molecule, "guess_bonds"):
            with contextlib.suppress(Exception):
                molecule.guess_bonds()
        molecule.set_atoms_id()
        return frozenset(tuple(sorted((bond.atom1.id, bond.atom2.id))) for bond in molecule.bonds)
