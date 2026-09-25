from __future__ import annotations

from typing import List, Optional, Tuple

from scm.plams import Settings

import scm.active_learning.journey_scheduler.simple_active_learning as simple_active_learning
import scm.active_learning.results.task as task_results_module
from scm.active_learning.checker_getters import DecoupledCheckerGetter
from scm.active_learning.checker_getters.checkers import AMSTrajChecker
from scm.active_learning.checker_getters.getters import AMSTrajGetter
from scm.active_learning.engines import AMSEngine, ASEEngine
from scm.active_learning.engines.atomic_container import SCMChemicalSystem
from scm.active_learning.journey_scheduler import SimpleActiveLearningJourney
from scm.active_learning.results import CheckGetResults, CollectionCheckGetResults, EngineCheckResult, FrameGetterResult
from scm.active_learning.results.task.ams_task import AMSTaskResult
from scm.active_learning.task_parallelization import AMSParallelMaxJobsGPU, AMSSerialStrategy
from scm.active_learning.tasks import AMSTask


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


def _make_check_results(success: bool) -> CollectionCheckGetResults:
    check = EngineCheckResult(
        engine_id="engine",
        task_id="task",
        system_id="system",
        checker_id="checker",
        property="energy",
        metric="mae",
        value=0.0,
        target=0.0,
        n_entries=1,
        success="OK" if success else "FAIL",
    )
    check_get = CheckGetResults(
        validity_results=[check],
        getter_results=FrameGetterResult(
            engine_id="engine",
            task_id="task",
            system_id="system",
            checker_id="checker",
            getter_id="getter",
        ),
    )
    return CollectionCheckGetResults(validity_get_results=[check_get])


def _make_accuracy_results(success: bool) -> list[EngineCheckResult]:
    return [
        _make_engine_check_result(
            success,
            checker_id="accuracy",
            property_name="energy",
        )
    ]


def _make_ams_task(task_id: str = "task", system_id: str = "sys-1") -> AMSTask:
    source_settings = Settings()
    source_settings.input.ams.task = "SinglePoint"
    return AMSTask(
        task_id=task_id,
        source_settings=source_settings.as_dict(),
        atomistic_system=SCMChemicalSystem(system_id=system_id, chemical_systems={"": "H"}),
    )


def test_simple_active_learning_journey_advance():
    journey = SimpleActiveLearningJourney.model_construct(
        checker_getter=None,
        task=None,
        steps=SimpleActiveLearningJourney.ListSteps(cumulative_values=[1, 2]),
        max_attempts=0,
        gluer=None,
        idx_=0,
        attempt_idx_=3,
        previous_job_=None,
    )
    validity = _make_check_results(True)
    accuracy = _make_accuracy_results(True)

    result = journey.advance(validity, accuracy)

    assert result.skip_training_reason == "All the tasks converged"
    assert journey.idx_ == 1
    assert journey.attempt_idx_ == 0
    assert journey.steps.md_nsteps == 2

    journey = SimpleActiveLearningJourney.model_construct(
        checker_getter=None,
        task=None,
        steps=SimpleActiveLearningJourney.ListSteps(cumulative_values=[2]),
        max_attempts=2,
        gluer=None,
        idx_=0,
        attempt_idx_=0,
        previous_job_=None,
    )

    journey.register_attempt()
    result = journey.advance(_make_check_results(False), accuracy)
    assert result.skip_training_reason == ""
    assert journey.idx_ == 0

    journey.register_attempt()
    result = journey.advance(_make_check_results(True), _make_accuracy_results(False))
    assert result.skip_training_reason == ""
    assert journey.idx_ == 1


def test_simple_active_learning_journey_init_and_table():
    checker = DecoupledCheckerGetter(checker=AMSTrajChecker(), getter=AMSTrajGetter())
    task = _make_ams_task(task_id="md", system_id="sys-1")
    journey = SimpleActiveLearningJourney(
        checker_getter=checker,
        task=task,
        steps=SimpleActiveLearningJourney.ListSteps(md_nsteps=20, cumulative_values=[10, 20]),
        max_attempts=2,
    )

    assert journey.n_steps() == 2
    # since we are now at the first step, therefore we left only one!
    assert journey.n_steps_left() == 1
    assert journey.is_finished() is False
    assert journey.idx_ == 0
    assert journey.attempt_idx_ == 0
    assert isinstance(journey.task_parallelization, AMSSerialStrategy)

    table = journey.journey_table()
    assert journey.start_separator in table
    assert journey.end_separator in table
    assert "md" in table
    assert "sys-1" in table
    assert "AMSTraj" in table


def test_simple_active_learning_current_batch_updates_steps():
    task = _make_ams_task(task_id="md", system_id="sys-1")
    journey = SimpleActiveLearningJourney.model_construct(
        checker_getter=None,
        task=task,
        steps=SimpleActiveLearningJourney.ListSteps(md_nsteps=15, cumulative_values=[5, 15]),
        max_attempts=1,
        gluer=None,
        idx_=1,
        attempt_idx_=0,
        previous_job_=None,
    )

    new_task: AMSTask = journey._current_batch_tasks()[0]
    assert new_task.settings.input.ams.MolecularDynamics.NSteps == 15


def test_simple_active_learning_advance_first_step_train_and_limits():
    journey = SimpleActiveLearningJourney.model_construct(
        checker_getter=None,
        task=None,
        steps=SimpleActiveLearningJourney.ListSteps(md_nsteps=2, cumulative_values=[1, 2]),
        max_attempts=2,
        gluer=None,
        first_step_train=True,
        idx_=0,
        attempt_idx_=0,
        previous_job_=None,
    )

    result = journey.advance(_make_collection_results([True]), _make_accuracy_results(True))
    assert result.skip_training_reason == ""
    assert journey.idx_ == 1
    assert journey.attempt_idx_ == 0

    journey = SimpleActiveLearningJourney.model_construct(
        checker_getter=None,
        task=None,
        steps=SimpleActiveLearningJourney.ListSteps(md_nsteps=1, cumulative_values=[1]),
        max_attempts=1,
        gluer=None,
        idx_=0,
        attempt_idx_=1,
        previous_job_=None,
    )
    result = journey.advance(_make_collection_results([True]), _make_accuracy_results(True))
    assert result.skip_training_reason == "All the tasks converged"
    assert journey.idx_ == 1


def test_simple_active_learning_finish_loop_on_max_attempts():
    journey = SimpleActiveLearningJourney.model_construct(
        checker_getter=None,
        task=None,
        steps=SimpleActiveLearningJourney.ListSteps(md_nsteps=3, cumulative_values=[1, 2, 3]),
        max_attempts=1,
        finish_loop_on_max_attempts=True,
        gluer=None,
        idx_=0,
        attempt_idx_=1,
        previous_job_=None,
    )

    result = journey.advance(_make_collection_results([False]), _make_accuracy_results(True))

    assert result.skip_training_reason == ""
    assert journey.idx_ == journey.n_steps()
    assert journey.is_finished() is True


class DummyResults:
    def __init__(self, rkf_path: str = "dummy/results.rkf", job: object | None = None):
        self.rkf_path = rkf_path
        self.job = job

    def rkfpath(self) -> str:
        return self.rkf_path


class DummyJob:
    def __init__(self):
        self.name = "dummy-job"
        self.settings = Settings()
        self.run_calls: List[Tuple[object | None, object | None]] = []

    def run(self, jobrunner=None, watch=None):
        self.run_calls.append((jobrunner, watch))
        return DummyResults(job=self)

    def get_runscript(self) -> str:
        return "dummy-runscript"


class DummyJobWithPath:
    def __init__(self, path: str):
        self.path = path


class DummyAMSTaskResult:
    def __init__(self, task, engine, plams_results):
        self.task = task
        self.engine = engine
        self.plams_results = plams_results

    @property
    def executed_job(self):
        return getattr(self.plams_results, "job", None)


class DummyGluer:
    def __init__(self):
        self.calls: List[Tuple[object, Optional[object]]] = []

    def run(self, current_job, previous_job=None):
        self.calls.append((current_job, previous_job))


class DummyCheckGetter:
    checker_id = "checker"
    getter_id = "getter"

    def run(self, result, **kwargs):
        check = _make_engine_check_result(True, task_id="task", system_id="sys", checker_id="checker")
        getter = FrameGetterResult(
            engine_id=check.engine_id,
            task_id="task",
            system_id="sys",
            checker_id="checker",
            getter_id="getter",
        )
        return CheckGetResults(validity_results=[check], getter_results=getter)


class DummyParallelization:
    def __init__(self, result):
        self.result = result
        self.calls: List[List[Tuple[object, object]]] = []

    def run_tasks(self, task_engine_coll):
        self.calls.append(task_engine_coll)
        return [self.result]


def test_simple_active_learning_run_current_batch_uses_gluer(monkeypatch):

    job = DummyJob()

    def fake_build_job(self, engine, extra=None):
        return job

    monkeypatch.setattr(AMSTask, "build_job", fake_build_job)
    monkeypatch.setattr(task_results_module, "AMSTaskResult", DummyAMSTaskResult)
    monkeypatch.setattr(simple_active_learning, "AMSTaskResult", DummyAMSTaskResult)

    gluer = DummyGluer()
    journey = SimpleActiveLearningJourney.model_construct(
        checker_getter=DummyCheckGetter(),
        task=_make_ams_task(task_id="task", system_id="sys"),
        steps=SimpleActiveLearningJourney.ListSteps(md_nsteps=1, cumulative_values=[1]),
        max_attempts=1,
        gluer=gluer,
        task_parallelization=AMSSerialStrategy(watch_ams_log_stdout=True),
        idx_=0,
        attempt_idx_=0,
        previous_job_="prev-job",
    )

    results = journey.run_current_batch(engine=AMSEngine(engine_id="engine"))

    assert journey.attempt_idx_ == 1
    assert journey.previous_job_ == "prev-job"
    assert journey._current_attempt_job is job
    assert job.run_calls == [(None, True)]
    assert gluer.calls == [(job, "prev-job")]
    assert len(results.validity_get_results) == 1


def test_simple_active_learning_run_current_batch_delegates_to_task_parallelization():
    job = DummyJobWithPath(path="delegated-job")
    task_result = AMSTaskResult.model_construct(
        task=_make_ams_task(task_id="task", system_id="sys"),
        engine=AMSEngine(engine_id="engine"),
        plams_results=DummyResults(job=job),
    )
    parallelization = DummyParallelization(task_result)
    journey = SimpleActiveLearningJourney.model_construct(
        checker_getter=DummyCheckGetter(),
        task=_make_ams_task(task_id="task", system_id="sys"),
        steps=SimpleActiveLearningJourney.ListSteps(md_nsteps=1, cumulative_values=[1]),
        max_attempts=1,
        gluer=None,
        task_parallelization=parallelization,
        idx_=0,
        attempt_idx_=0,
        previous_job_=None,
    )

    results = journey.run_current_batch(engine=AMSEngine(engine_id="engine"))

    assert journey.attempt_idx_ == 1
    assert journey._current_attempt_job is job
    assert len(parallelization.calls) == 1
    assert len(parallelization.calls[0]) == 1
    assert len(results.validity_get_results) == 1


def test_ams_parallel_gpu_strategy_applies_runscript_settings_to_ase_engine(monkeypatch):
    task = _make_ams_task(task_id="task", system_id="sys")
    engine = ASEEngine(engine_id="ase", calculator="tests.test_journey_simple_active_learning.DummyCalculator")
    strategy = AMSParallelMaxJobsGPU()
    captured = {}
    job = DummyJob()

    def fake_build_job(self, engine, extra=None):
        captured["engine"] = engine
        captured["extra"] = extra
        return job

    def fail_if_serial(self, engine, *args, **kwargs):
        raise AssertionError("AMS task should not fall back to the serial task.run() path for ASEEngine")

    monkeypatch.setattr(AMSTask, "build_job", fake_build_job)
    monkeypatch.setattr(AMSTask, "run", fail_if_serial)
    monkeypatch.setattr(task_results_module, "AMSTaskResult", DummyAMSTaskResult)

    results = strategy.run_tasks([(task, engine)])

    assert len(results) == 1
    assert isinstance(captured["engine"], AMSEngine)
    assert captured["extra"].runscript.nproc == 1
    assert captured["extra"].runscript.preamble_lines == ["export OMP_NUM_THREADS=1"]
    assert job.run_calls == [(None, False)]


def test_simple_active_learning_build_results_updates_previous_job_on_success_only():
    journey = SimpleActiveLearningJourney.model_construct(
        checker_getter=None,
        task=None,
        steps=SimpleActiveLearningJourney.ListSteps(md_nsteps=2, cumulative_values=[1, 2]),
        max_attempts=2,
        gluer=None,
        idx_=0,
        attempt_idx_=1,
        previous_job_=None,
    )

    old_job = DummyJobWithPath(path="old-path")
    new_job = DummyJobWithPath(path="new-path")

    journey.previous_job_ = old_job
    journey._current_attempt_job = new_job
    success_result = journey.build_results("ok", finished=True, success=True)

    assert journey.previous_job_ is new_job
    assert journey._current_attempt_job is None
    assert journey.idx_ == 1
    assert journey.attempt_idx_ == 0
    assert success_result.info["previous_job"] == "new-path"
    assert success_result.info["success"] is True

    journey.previous_job_ = old_job
    journey._current_attempt_job = new_job
    failed_result = journey.build_results("failed", finished=False, success=False)

    assert journey.previous_job_ is old_job
    assert journey._current_attempt_job is None
    assert journey.idx_ == 1
    assert journey.attempt_idx_ == 0
    assert failed_result.info["previous_job"] == "old-path"
    assert failed_result.info["success"] is False
