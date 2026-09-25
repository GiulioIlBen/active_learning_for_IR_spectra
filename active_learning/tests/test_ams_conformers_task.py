from __future__ import annotations

import pytest

pytest.importorskip("scm.conformers", reason="requires optional dependency scm.conformers")

from scm.plams import Settings

import scm.active_learning.results.task as task_results_module
from scm.active_learning.engines import AMSEngine
from scm.active_learning.task_parallelization import AMSParallelCPU
from scm.active_learning.tasks.ams_conformers_task import AMSConformersTask


def test_ams_conformers_task_type_default():
    assert AMSConformersTask.model_fields["type"].default == "AMSConformersTask"


def test_ams_parallel_cpu_applies_nproc_to_conformers(monkeypatch):
    task = AMSConformersTask.model_construct()
    engine = AMSEngine(engine_id="engine")
    captured = {}

    class DummyJob:
        name = "conformers"

        def __init__(self, settings):
            self.settings = settings

        def get_runscript(self):
            return "conformers"

        def run(self, jobrunner=None, watch=False):
            captured["run"] = (jobrunner, watch)
            return "results"

    class DummyConformersResults:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def fake_build_job(self, engine, extra=None):
        captured["engine"] = engine
        captured["extra"] = extra
        return DummyJob(extra or Settings())

    monkeypatch.setattr(AMSConformersTask, "build_job", fake_build_job)
    monkeypatch.setattr(task_results_module, "AMSConformersResults", DummyConformersResults)

    results = AMSParallelCPU(nproc=1).run_tasks([(task, engine)])

    assert captured["extra"].runscript.nproc == 1
    assert captured["run"] == (None, False)
    assert isinstance(results[0], DummyConformersResults)
