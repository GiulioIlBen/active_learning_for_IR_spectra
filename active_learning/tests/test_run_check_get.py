from __future__ import annotations

from typing import ClassVar, List, Tuple, Type

from pydantic import BaseModel

from scm.active_learning.checker_getters import DecoupledCheckerGetter
from scm.active_learning.checker_getters.checkers.core import Checker
from scm.active_learning.engines import Engine
from scm.active_learning.journey_scheduler import SequentialStepsJourney
from scm.active_learning.results import EngineCheckResult, FrameGetterResult, TaskResult
from scm.active_learning.task_parallelization import SerialStrategy
from scm.active_learning.tasks import Task


class DummyTask(Task[Engine]):
    task_id: str
    order_log: List[str]

    @property
    def system_id(self) -> str:
        return "sys1"

    def run(self, engine: Engine, *args, **kwargs) -> "DummyTaskResult":
        self.order_log.append("task")
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


class DummyChecker(Checker[DummyTaskResult]):
    supported_tasks: ClassVar[Tuple[Type[Task]]] = (DummyTask,)
    checker_id: str
    order_log: List[str]

    def run(self, result: DummyTaskResult, **kwargs) -> List[EngineCheckResult]:
        self.order_log.append("check")
        return [
            EngineCheckResult(
                engine_id=result.from_engine.engine_id,
                task_id=result.from_task.task_id,
                system_id=result.from_task.system_id,
                checker_id=self.checker_id,
                property="energy",
                metric="mae",
                value=1.0,
                target=0.0,
                n_entries=1,
                success="FAIL",
            )
        ]


class DummyGetter(BaseModel):
    getter_id: str
    order_log: List[str]

    def run(self, result: DummyTaskResult, checker_results: List[EngineCheckResult], **kwargs) -> FrameGetterResult:
        self.order_log.append("get")
        return FrameGetterResult(
            engine_id=result.engine.engine_id,
            task_id=result.from_task.task_id,
            system_id=result.from_task.system_id,
            checker_id=checker_results[-1].checker_id,
            getter_id=self.getter_id,
        )


def test_run_check_get_sequence():
    order_log: List[str] = []
    engine = Engine(engine_id="engine1")
    task = DummyTask.model_construct(task_id="task1", order_log=order_log)
    checker = DummyChecker.model_construct(checker_id="checker1", order_log=order_log)
    getter = DummyGetter.model_construct(getter_id="getter1", order_log=order_log)

    check_getter = DecoupledCheckerGetter.model_construct(
        checker_id="checkget1",
        checker=checker,
        getter=getter,
    )
    journey = SequentialStepsJourney.model_construct(
        couples=[(task, check_getter)],
        max_attempts=0,
        task_parallelization=SerialStrategy(),
    )

    results = journey.run_current_batch(engine=engine)
    assert order_log == ["task", "check", "get"]
    print("---")
    print(results.validity_results_table())
    print("---")
    print(results.getter_results_table())
