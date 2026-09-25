from __future__ import annotations

from types import SimpleNamespace

from scm.moliterate.analysis import PairwiseResult

from scm.active_learning.callbacks.al_minimal_logger import ALMinimalLogger
from scm.active_learning.callbacks.al_state_logger import ALStateLogger, check_res_table
from scm.active_learning.loop.trainer_check import TrainerCheckResult
from scm.active_learning.results import EngineCheckResult


class DummyJourney:
    def journey_table(self):
        return "journey"

    def history_table(self):
        return "history"


class DummyLoop:
    def __init__(self):
        self.journey = DummyJourney()


class DummyTaskResults:
    def __init__(self, validity_get_results):
        self.validity_get_results = validity_get_results

    def getter_results_table(self):
        return "getter-results"


def _engine_check_result():
    return EngineCheckResult(
        engine_id="engine",
        task_id="task",
        system_id="sys",
        checker_id="checker",
        property="energy",
        metric="rmse",
        value=1.0,
        n_entries=1,
        success="OK",
    )


def _pairwise_result():
    return PairwiseResult(
        property="energy",
        metric="rmse",
        units="eV",
        value=1.0,
        target=2.0,
        n_entries=1,
        success="OK",
    )


def test_check_res_table_none_returns_empty():
    assert check_res_table(None) == ""


def test_al_state_logger_skips_when_disabled_or_missing_results(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "scm.active_learning.callbacks.al_state_logger.log_level",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    logger = ALStateLogger(log_journey_table=False)
    loop = DummyLoop()
    state = SimpleNamespace(iteration_al=1, task_results=None, accuracy_checks_results=None)

    logger.before_loop(loop)
    logger.after_convergence(loop, state)
    logger.after_run_task(loop, state)
    logger.after_accuracy(loop, state)

    assert calls == []


def test_al_state_logger_emits_tables(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "scm.active_learning.callbacks.al_state_logger.log_level",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    loop = DummyLoop()
    logger = ALStateLogger(log_journey_table=True)
    validity = [SimpleNamespace(validity_results=[_engine_check_result()])]
    state = SimpleNamespace(
        iteration_al=2,
        task_results=DummyTaskResults(validity_get_results=validity),
        accuracy_checks_results=[_engine_check_result()],
    )

    logger.before_loop(loop)
    logger.after_run_task(loop, state)
    logger.after_accuracy(loop, state)
    logger.after_convergence(loop, state)

    assert len(calls) == 3


def test_al_minimal_logger_logs_explanatory_empty_history(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "scm.active_learning.callbacks.al_minimal_logger.log_level",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    class ExplainingJourney:
        def history_table(self):
            return (
                "==== MoleculesJourney History ====\n"
                "No journey history has been recorded yet.\n"
                "Tasks are active, but convergence/advance has not produced a summary snapshot for this iteration."
            )

    loop = SimpleNamespace(journey=ExplainingJourney())
    state = SimpleNamespace(iteration_al=0)

    ALMinimalLogger().after_train(loop, state)

    assert len(calls) == 1
    assert "No journey history has been recorded yet." in calls[0][1]["history_table"]
