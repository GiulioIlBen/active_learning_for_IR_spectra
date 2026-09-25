from __future__ import annotations

from typing import Any, List

from ase import Atoms
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData

from scm.active_learning.checker_getters import UnionCheckerGetter, UnionMember
from scm.active_learning.engines import Engine
from scm.active_learning.results import EngineCheckResult, FrameGetterResult
from scm.active_learning.results.task.core import TaskResult
from scm.active_learning.tasks import Task


class DummyTask(Task[Engine]):
    task_id: str = "go"

    @property
    def system_id(self) -> str:
        return "M0000"

    def run(self, engine: Engine, *args, **kwargs):
        raise NotImplementedError


class DummyTaskResult(TaskResult):
    task: DummyTask
    engine: Engine

    @property
    def from_task(self) -> Task:
        return self.task

    @property
    def from_engine(self) -> Engine:
        return self.engine


class FakeMember(UnionMember):
    check_getter: Any = None
    name: str
    success: bool = True
    n_frames: int = 1

    @property
    def checker_id(self) -> str:
        return f"{self.name}Checker"

    @property
    def getter_id(self) -> str:
        return self.name

    def check(self, result: TaskResult) -> List[EngineCheckResult]:
        return [
            EngineCheckResult(
                engine_id="engine",
                task_id="go",
                system_id="M0000",
                checker_id=f"{self.name}Checker",
                property="p",
                metric="m",
                value=0.0,
                target=0.0,
                n_entries=1,
                success="OK" if self.success else "FAIL",
            )
        ]

    def get(self, result: TaskResult, checks: List[EngineCheckResult]) -> FrameGetterResult:
        dataset = InMemoryMolData()
        for _ in range(self.n_frames):
            dataset.add_system(ChemDataEntry(system=Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])))
        return FrameGetterResult(
            engine_id="engine",
            task_id="go",
            system_id="M0000",
            checker_id=f"{self.name}Checker",
            getter_id=self.name,
            dataset=dataset,
        )


def _result() -> DummyTaskResult:
    return DummyTaskResult(task=DummyTask(), engine=Engine(engine_id="engine"))


def _getter_ids(union: UnionCheckerGetter) -> list[str]:
    res = union.run(result=_result())
    return [row.metadata["getter_id"] for row in res.getter_results.get_dataset_with_metadata()]


def test_union_merges_frames_and_keeps_each_getter_id():
    union = UnionCheckerGetter(
        members=[
            FakeMember(name="GOTraj", n_frames=2),
            FakeMember(name="NMS", n_frames=5, run_getter="if_checks_pass"),
        ]
    )

    assert _getter_ids(union) == ["GOTraj"] * 2 + ["NMS"] * 5


def test_union_skips_gated_getter_when_an_earlier_check_fails():
    union = UnionCheckerGetter(
        members=[
            FakeMember(name="GOTraj", n_frames=2, success=False),
            FakeMember(name="NMS", n_frames=5, run_getter="if_checks_pass"),
        ]
    )

    res = union.run(result=_result())

    assert _getter_ids(union) == ["GOTraj"] * 2
    assert len(res.validity_results) == 2
    assert not all(check.is_success() for check in res.validity_results)


def test_union_gate_ignores_checks_of_later_members():
    union = UnionCheckerGetter(
        members=[
            FakeMember(name="GOTraj", n_frames=2),
            FakeMember(name="NMS", n_frames=5, run_getter="if_checks_pass"),
            FakeMember(name="IR", n_frames=0, success=False),
        ]
    )

    assert _getter_ids(union) == ["GOTraj"] * 2 + ["NMS"] * 5


def test_never_member_contributes_checks_but_no_frames():
    union = UnionCheckerGetter(
        members=[
            FakeMember(name="GOTraj", n_frames=2),
            FakeMember(name="IR", n_frames=3, success=False, run_getter="never"),
        ]
    )

    res = union.run(result=_result())

    assert _getter_ids(union) == ["GOTraj"] * 2
    assert not all(check.is_success() for check in res.validity_results)
