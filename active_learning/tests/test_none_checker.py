from __future__ import annotations

from pydantic import TypeAdapter

from scm.active_learning.checker_getters.checkers import ConcreteChecker, NoneChecker
from scm.active_learning.checker_getters.decoupled_check_getter import DecoupledCheckerGetter
from scm.active_learning.engines import Engine
from scm.active_learning.results.task.core import TaskResult
from scm.active_learning.tasks import Task


class DummyTask(Task[Engine]):
    task_id: str

    @property
    def system_id(self) -> str:
        return "sys"

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


def test_none_checker_run_returns_no_results():
    checker = NoneChecker()
    result = DummyTaskResult(
        task=DummyTask(task_id="task"),
        engine=Engine(engine_id="engine"),
    )

    assert len(checker.run(result=result)) == 1


def test_none_checker_is_registered_as_concrete_checker():
    adapter = TypeAdapter(ConcreteChecker)
    checker = adapter.validate_python({"type": "NoneChecker"})

    assert isinstance(checker, NoneChecker)
    assert checker.checker_id == "NoneChecker"


def test_none_checker_validates_in_decoupled_checker_getter():
    check_getter = DecoupledCheckerGetter.model_validate(
        {
            "type": "DecoupledCheckerGetter",
            "checker": {"type": "NoneChecker"},
            "getter": {"type": "AMSTrajGetter"},
        }
    )

    assert isinstance(check_getter.checker, NoneChecker)
