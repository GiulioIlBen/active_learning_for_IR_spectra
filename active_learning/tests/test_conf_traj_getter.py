from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter

from scm.active_learning.checker_getters import ConcreteCheckerGetter
from scm.active_learning.checker_getters.checkers import ConfJobFailChecker
from scm.active_learning.checker_getters.getters import (
    AMSTrajGetter,
    ConformersGetter,
    ConfTrajGetter,
)
from scm.active_learning.results import FrameGetterResult

WORKER_POOL_CLEANUP_ERROR = """Traceback (most recent call last):
  File "/ams/python/lib/python3.8/weakref.py", line 642, in _exitfunc
    f()
  File "/ams/python/lib/python3.8/shutil.py", line 709, in rmtree
    onerror(os.lstat, path, sys.exc_info())
FileNotFoundError: [Errno 2] No such file or directory: '/jobs/conf/awp_0_3nsrc8ze'"""
SED_PERMISSION_WARNING = (
    "sed: preserving permissions for '/ams/python/AMS2026.1.venv/sedAr7iaa': Operation not supported"
)


class DummyJob:
    def __init__(self, path, error_message=""):
        self.name = "engine-system-conf"
        self.path = path
        self.error_message = error_message

    def get_errormsg(self):
        return self.error_message


class DummyResults:
    def __init__(self, job, rkf_path, file_error="", file_output=""):
        self.job = job
        self.rkf_path = rkf_path
        self.file_error = file_error
        self.file_output = file_output
        self.read_file_calls = []
        self.grep_output_calls = []

    def rkfpath(self):
        return self.rkf_path

    def read_file(self, filename):
        self.read_file_calls.append(filename)
        return {"$JN.err": self.file_error}[filename]

    def grep_output(self, pattern, options=""):
        self.grep_output_calls.append((pattern, options))
        return [line for line in self.file_output.splitlines() if pattern in line]


def _result(job, rkf_path="conformers.rkf", file_error="", file_output=""):
    return SimpleNamespace(
        plams_results=DummyResults(
            job=job,
            rkf_path=rkf_path,
            file_error=file_error,
            file_output=file_output,
        ),
        from_task=SimpleNamespace(task_id="conf", system_id="M0000"),
        from_engine=SimpleNamespace(engine_id="MACE_EFD00"),
    )


@pytest.mark.parametrize(
    ("plams_error", "file_error", "file_output", "handled_error"),
    [
        (
            "Geometry optimization failed! (Did not converge.)",
            "",
            "",
            "Geometry optimization failed! (Did not converge.)",
        ),
        (
            "Could not determine error message.",
            "Major bond changes after geometry optimization of input conformer.",
            "",
            "Major bond changes after geometry optimization of input conformer.",
        ),
        (
            "",
            "NORMAL TERMINATION with errors",
            "ERROR: Geometry optimization failed! (Did not converge.)",
            "Geometry optimization failed! (Did not converge.)",
        ),
    ],
)
def test_conf_job_fail_checker_handles_known_error_messages(
    monkeypatch,
    tmp_path,
    plams_error,
    file_error,
    file_output,
    handled_error,
):
    job = DummyJob(path=tmp_path, error_message=plams_error)

    def unexpected_load(_):
        raise AssertionError("The failed conformer set should not be loaded for a handled error.")

    monkeypatch.setattr(
        "scm.active_learning.checker_getters.checkers.conformers_energy_checker.load_dataset",
        unexpected_load,
    )

    result = _result(job, file_error=file_error, file_output=file_output)
    results = ConfJobFailChecker().run(result)

    assert len(results) == 1
    assert results[0].success == "GONoConv"
    assert handled_error in results[0].value
    assert result.plams_results.read_file_calls == ["$JN.err"]
    assert result.plams_results.grep_output_calls == [
        ("Geometry optimization failed! (Did not converge.)", "-F"),
        ("Major bond changes after geometry optimization of input conformer.", "-F"),
    ]


def test_conf_job_fail_checker_accepts_successful_conformer_set(monkeypatch, tmp_path):
    job = DummyJob(path=tmp_path)
    monkeypatch.setattr(
        "scm.active_learning.checker_getters.checkers.conformers_energy_checker.load_dataset",
        lambda _: object(),
    )

    results = ConfJobFailChecker().run(_result(job, file_output="NORMAL TERMINATION"))

    assert len(results) == 1
    assert results[0].success == "OK"
    assert results[0].value == ""


def test_conf_job_fail_checker_ignores_configured_non_fatal_error(monkeypatch, tmp_path):
    job = DummyJob(path=tmp_path)
    monkeypatch.setattr(
        "scm.active_learning.checker_getters.checkers.conformers_energy_checker.load_dataset",
        lambda _: object(),
    )

    results = ConfJobFailChecker(ignored_error_strings=["sed: preserving permissions"]).run(
        _result(job, file_error=SED_PERMISSION_WARNING, file_output="NORMAL TERMINATION")
    )

    assert len(results) == 1
    assert results[0].success == "OK"
    assert results[0].value == SED_PERMISSION_WARNING


def test_conf_job_fail_checker_defers_worker_pool_cleanup_error_by_default(monkeypatch, tmp_path):
    job = DummyJob(path=tmp_path)
    monkeypatch.setattr(
        "scm.active_learning.checker_getters.checkers.conformers_energy_checker.load_dataset",
        lambda _: object(),
    )

    results = ConfJobFailChecker().run(_result(job, file_error=WORKER_POOL_CLEANUP_ERROR))

    assert results[0].success == "OK"
    assert results[0].value == ""


def test_conf_job_fail_checker_collects_only_worker_pool_cleanup_error(tmp_path):
    job = DummyJob(path=tmp_path)
    checker = ConfJobFailChecker(
        collect_geometry_errors=False,
        collect_worker_pool_cleanup_error=True,
    )

    results = checker.run(_result(job, file_error=WORKER_POOL_CLEANUP_ERROR))

    assert results[0].success == "AWPCleanupError"
    assert results[0].value == WORKER_POOL_CLEANUP_ERROR


def test_conf_job_fail_checker_raises_for_unhandled_error(tmp_path):
    job = DummyJob(path=tmp_path)

    with pytest.raises(RuntimeError, match="Unhandled conformer job error"):
        ConfJobFailChecker().run(_result(job, file_error="Unexpected conformer generator failure."))


def test_conf_job_fail_checker_raises_for_unhandled_rkf_failure(monkeypatch, tmp_path):
    job = DummyJob(path=tmp_path)

    def fail_load(_):
        raise ValueError("Missing History")

    monkeypatch.setattr(
        "scm.active_learning.checker_getters.checkers.conformers_energy_checker.load_dataset",
        fail_load,
    )

    with pytest.raises(RuntimeError, match="Unhandled conformer results failure"):
        ConfJobFailChecker().run(_result(job))


def test_conf_traj_getter_delegates_to_nested_ams_getter(monkeypatch, tmp_path):
    trajectory_path = tmp_path / "initGO" / "ams.rkf"
    trajectory_path.parent.mkdir()
    trajectory_path.touch()
    calls = {}

    def fake_run_from_rkf_path(self, result, checker_results, rkf_path, getter_id=None, **kwargs):
        calls.update(
            nested_getter=self,
            result=result,
            checker_results=checker_results,
            rkf_path=rkf_path,
            getter_id=getter_id,
        )
        return FrameGetterResult(
            engine_id=result.from_engine.engine_id,
            task_id=result.from_task.task_id,
            system_id=result.from_task.system_id,
            checker_id="ConfJFail",
            getter_id=getter_id,
        )

    monkeypatch.setattr(AMSTrajGetter, "run_from_rkf_path", fake_run_from_rkf_path)
    nested_getter = AMSTrajGetter(condition_filters=[])
    getter = ConfTrajGetter(job=nested_getter)
    result = _result(DummyJob(path=tmp_path))

    getter_result = getter.run(result, checker_results=[])

    assert calls["nested_getter"].condition_filters == []
    assert calls["result"] is result
    assert calls["rkf_path"] == trajectory_path
    assert calls["getter_id"] == "ConfTrajGetter"
    assert getter_result.getter_id == "ConfTrajGetter"


def test_conf_traj_getter_raises_when_init_go_trajectory_is_missing(tmp_path):
    getter = ConfTrajGetter()
    result = _result(DummyJob(path=tmp_path))

    with pytest.raises(FileNotFoundError, match="initGO/ams.rkf"):
        getter.run(result, checker_results=[])


def test_conformer_concat_checker_getter_round_trip():
    checker_getter = (ConfJobFailChecker() + ConfTrajGetter(job=AMSTrajGetter())) >> (
        ConfJobFailChecker(
            collect_geometry_errors=False,
            collect_worker_pool_cleanup_error=True,
        )
        + ConformersGetter()
    )

    restored = TypeAdapter(ConcreteCheckerGetter).validate_python(checker_getter.model_dump())

    assert len(restored.check_getters) == 2
    assert isinstance(restored.check_getters[0].getter, ConfTrajGetter)
    assert isinstance(restored.check_getters[0].getter.job, AMSTrajGetter)
    final_checker = restored.check_getters[1].checker
    assert isinstance(final_checker, ConfJobFailChecker)
    assert not final_checker.collect_geometry_errors
    assert final_checker.collect_worker_pool_cleanup_error
    assert isinstance(restored.check_getters[1].getter, ConformersGetter)
