from __future__ import annotations

from typing import List, Optional

from scm.active_learning.checker_getters import CheckerGetter
from scm.active_learning.engines import Engine
from scm.active_learning.journey_scheduler import (
    JourneyScheduler,
    SequentialStepsJourney,
)
from scm.active_learning.results import (
    CheckGetResults,
    CollectionCheckGetResults,
    EngineCheckResult,
    FrameGetterResult,
    JourneyAdvanceResult,
    TaskResult,
)
from scm.active_learning.task_parallelization import SerialStrategy
from scm.active_learning.tasks import Task


def _make_engine_check_result(
    success: bool,
    *,
    engine_id: str = "engine",
    task_id: str = "task",
    system_id: str = "system",
    checker_id: str = "checker",
    property_name: str = "energy",
) -> EngineCheckResult:
    return EngineCheckResult(
        engine_id=engine_id,
        task_id=task_id,
        system_id=system_id,
        checker_id=checker_id,
        property=property_name,
        metric="mae",
        value=0.0,
        target=0.0,
        n_entries=1,
        success="OK" if success else "FAIL",
    )


def _make_check_get_results(success: bool, *, task_id: str = "task", system_id: str = "system") -> CheckGetResults:
    check = _make_engine_check_result(success, task_id=task_id, system_id=system_id)
    getter = FrameGetterResult(
        engine_id=check.engine_id,
        task_id=task_id,
        system_id=system_id,
        checker_id=check.checker_id,
        getter_id="getter",
    )
    return CheckGetResults(validity_results=[check], getter_results=getter)


def _make_collection_results(successes: List[bool]) -> CollectionCheckGetResults:
    return CollectionCheckGetResults(validity_get_results=[_make_check_get_results(s) for s in successes])


def _make_accuracy_results(success: bool) -> list[EngineCheckResult]:
    return [
        _make_engine_check_result(
            success,
            checker_id="accuracy",
            property_name="energy",
        )
    ]


class DummyTask(Task[Engine]):
    task_id: str
    order_log: List[str]

    @property
    def system_id(self) -> str:
        return "sys1"

    def run(self, engine: Engine, *args, **kwargs) -> "DummyTaskResult":
        self.order_log.append(f"task:{self.task_id}")
        return DummyTaskResult(task=self, engine=engine)


class DummyTaskResult(TaskResult):
    task: DummyTask
    engine: Engine

    @property
    def from_task(self) -> Task:
        return self.task

    @property
    def from_engine(self) -> Engine:
        return self.engine


class DummyCheckerGetter(CheckerGetter):
    order_log: List[str]
    checker_id_: str
    getter_id_: str

    @property
    def checker_id(self):
        return self.checker_id_

    @property
    def getter_id(self):
        return self.getter_id_

    def run(self, result: TaskResult, c_res: Optional[List[EngineCheckResult]] = None, **kwargs) -> CheckGetResults:
        self.order_log.append(f"check:{self.checker_id}")
        checks = c_res or [
            _make_engine_check_result(
                True,
                engine_id=result.from_engine.engine_id,
                task_id=result.from_task.task_id,
                system_id=result.from_task.system_id,
                checker_id=self.checker_id,
            )
        ]
        getter = FrameGetterResult(
            engine_id=result.from_engine.engine_id,
            task_id=result.from_task.task_id,
            system_id=result.from_task.system_id,
            checker_id=self.checker_id,
            getter_id=self.getter_id,
        )
        return CheckGetResults(validity_results=checks, getter_results=getter)


class DummyJourney(JourneyScheduler):
    tasks: List[DummyTask]
    checkers: List[DummyCheckerGetter]
    order_log: List[str]
    attempts: int = 0

    @property
    def max_n_steps(self):
        return -1

    def restart_journey(self):
        self.attempts = 0

    def n_steps(self) -> int:
        return len(self.tasks)

    def n_steps_left(self) -> int:
        return len(self.tasks)

    def is_finished(self) -> bool:
        return False

    def register_attempt(self) -> None:
        self.attempts += 1
        self.order_log.append("attempt")

    def _current_batch_tasks(self) -> List[DummyTask]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.tasks

    def _current_batch_check_getters(self) -> List[DummyCheckerGetter]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.checkers

    def advance(
        self,
        validity_results: CollectionCheckGetResults,
        accuracy_results: List[EngineCheckResult],
    ) -> JourneyAdvanceResult:
        return JourneyAdvanceResult()


def test_journey_scheduler_run_current_batch_and_table():
    order_log: List[str] = []
    tasks = [
        DummyTask.model_construct(task_id="t1", order_log=order_log),
        DummyTask.model_construct(task_id="t2", order_log=order_log),
    ]
    checkers = [
        DummyCheckerGetter.model_construct(checker_id_="c1", getter_id_="g1", order_log=order_log),
        DummyCheckerGetter.model_construct(checker_id_="c2", getter_id_="g2", order_log=order_log),
    ]
    journey = DummyJourney.model_construct(
        tasks=tasks,
        checkers=checkers,
        order_log=order_log,
        task_parallelization=SerialStrategy(),
    )

    results = journey.run_current_batch(engine=Engine(engine_id="engine1"))  # pyright: ignore[reportArgumentType]

    assert order_log == ["attempt", "task:t1", "task:t2", "check:c1", "check:c2"]
    assert len(results.validity_get_results) == 2
    assert journey.attempts == 1


def test_sequential_steps_journey_current_batch_and_register_attempt():
    order_log: List[str] = []
    task1 = DummyTask.model_construct(task_id="t1", order_log=order_log)
    task2 = DummyTask.model_construct(task_id="t2", order_log=order_log)
    checker1 = DummyCheckerGetter.model_construct(checker_id_="c1", getter_id_="g1", order_log=order_log)
    checker2 = DummyCheckerGetter.model_construct(checker_id_="c2", getter_id_="g2", order_log=order_log)
    journey = SequentialStepsJourney.model_construct(
        couples=[(task1, checker1), (task2, checker2)],
        max_attempts=2,
        idx_=1,
        attempt_idx_=0,
        task_parallelization=SerialStrategy(),
    )

    assert journey._current_batch_tasks() == [task2]
    assert journey._current_batch_check_getters() == [checker2]
    journey.register_attempt()
    assert journey.attempt_idx_ == 1


def test_sequential_steps_journey_advance_success_and_failures():
    task = DummyTask.model_construct(task_id="t1", order_log=[])
    checker = DummyCheckerGetter.model_construct(checker_id_="c1", getter_id_="g1", order_log=[])

    journey = SequentialStepsJourney.model_construct(
        couples=[(task, checker)],
        max_attempts=0,
        _idx=0,
        _attempt_idx=3,
    )

    result = journey.advance(_make_collection_results([True]), _make_accuracy_results(True))
    assert result.skip_training_reason == "All the tasks converged"
    assert journey.idx_ == 1
    assert journey.attempt_idx_ == 0

    journey = SequentialStepsJourney.model_construct(couples=[(task, checker)], max_attempts=0, _idx=0, _attempt_idx=0)
    result = journey.advance(_make_collection_results([False]), _make_accuracy_results(True))
    assert result.skip_training_reason == ""
    assert journey.idx_ == 0

    result = journey.advance(_make_collection_results([True]), _make_accuracy_results(False))
    assert result.skip_training_reason == ""
    assert journey.idx_ == 0


def test_sequential_steps_journey_attempt_gate_and_table():
    task = DummyTask.model_construct(task_id="t1", order_log=[])
    checker = DummyCheckerGetter.model_construct(checker_id_="c1", getter_id_="g1", order_log=[])
    journey = SequentialStepsJourney.model_construct(couples=[(task, checker)], max_attempts=5, _idx=0, _attempt_idx=0)

    result = journey.advance(_make_collection_results([True]), _make_accuracy_results(True))
    assert result.skip_training_reason == ""
    assert journey.idx_ == 0

    table = journey.journey_table()
    assert journey.start_separator in table
    assert journey.end_separator in table
    assert "t1" in table
    assert "c1" in table
