from __future__ import annotations

import pytest

from scm.active_learning.checker_getters.ok_checker import OkCheckerGetter
from scm.active_learning.engines import Engine
from scm.active_learning.results import CheckGetResults
from scm.active_learning.results.frame_getter import FrameGetterResult
from scm.active_learning.results.task.core import TaskResult
from scm.active_learning.tasks import Task


class OkCheckerGetterWithDefault(OkCheckerGetter):
    pass


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


def test_ok_checker_getter_checker_id_property():
    getter = OkCheckerGetter(checker_id_="checker", getter_id_="getter")

    assert getter.checker_id == "checker"


def test_ok_checker_getter_run_uses_default_getter_res():
    getter = OkCheckerGetterWithDefault(checker_id_="checker", getter_id_="getter")
    result = DummyTaskResult(
        task=DummyTask(task_id="task"),
        engine=Engine(engine_id="engine"),
    )

    res = getter.run(result=result, c_res=[])

    assert isinstance(res, CheckGetResults)
    assert res.validity_results == []
    assert res.getter_results.engine_id == "engine"
    assert res.getter_results.task_id == "task"
    assert res.getter_results.system_id == "sys"
    assert res.getter_results == FrameGetterResult(
        engine_id="engine",
        task_id="task",
        system_id="sys",
        checker_id="checker",
        getter_id="getter",
    )


def test_ok_checker_getter_run_missing_default_getter_res():
    getter = OkCheckerGetter(checker_id_="checker", getter_id_="getter")

    with pytest.raises(AttributeError):
        getter.run(result=object())
