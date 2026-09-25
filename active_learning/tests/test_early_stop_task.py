from __future__ import annotations

from types import SimpleNamespace

from scm.active_learning.callbacks.early_stop_task import EarlyStopAccuracy, EarlyStopValidity
from scm.active_learning.loop.iteration_phase import IterPhase
from scm.active_learning.results import EngineCheckResult


def _check(
    value,
    *,
    success="OK",
    engine_id="eng",
    property="energy",
    metric="rmse",
    lower_is_better=True,
    value_type="float",
):
    return EngineCheckResult(
        engine_id=engine_id,
        task_id="task",
        system_id="sys",
        checker_id="chk",
        property=property,
        metric=metric,
        value=value,
        value_type=value_type,
        target="" if value_type == "str" else 0.0,
        lower_is_better=lower_is_better,
        n_entries=1,
        success=success,
    )


def _task_results(*checks):
    return SimpleNamespace(validity_get_results=[SimpleNamespace(validity_results=list(checks))])


def test_early_stop_skips_without_required_state():
    cb = EarlyStopAccuracy()
    state = SimpleNamespace(task_results=None, accuracy_checks_results=None, al_finished_reason=None)
    loop = SimpleNamespace(result=None)

    cb.after_convergence(loop, state)

    assert state.al_finished_reason is None


def test_early_stop_resets_when_all_selected_checks_pass():
    cb = EarlyStopAccuracy(patience=2)
    cb._no_improve_steps = 1
    cb._info_history = ["Errors| old"]

    prev_iter = SimpleNamespace(accuracy_checks_results=[_check(2.0, success="FAIL")])
    loop = SimpleNamespace(result=SimpleNamespace(iterations=[prev_iter]))
    state = SimpleNamespace(
        task_results=SimpleNamespace(validity_get_results=[]),
        accuracy_checks_results=[_check(1.0, success="OK")],
        al_finished_reason=None,
    )

    cb.after_convergence(loop, state)

    assert cb._no_improve_steps == 0
    assert cb._info_history == []
    assert state.al_finished_reason is None


def test_early_stop_resets_when_failed_value_improves():
    cb = EarlyStopAccuracy(patience=2)
    cb._no_improve_steps = 1

    prev_iter = SimpleNamespace(accuracy_checks_results=[_check(2.0, success="FAIL", engine_id="eng00")])
    loop = SimpleNamespace(result=SimpleNamespace(iterations=[prev_iter]))
    state = SimpleNamespace(
        task_results=SimpleNamespace(validity_get_results=[]),
        accuracy_checks_results=[_check(1.0, success="FAIL", engine_id="eng01")],
        al_finished_reason=None,
    )

    cb.after_convergence(loop, state)

    assert cb._no_improve_steps == 0
    assert cb._info_history == []


def test_early_stop_resets_when_number_of_failed_checks_reduces():
    cb = EarlyStopAccuracy(patience=2)
    cb._no_improve_steps = 1

    prev_iter = SimpleNamespace(
        accuracy_checks_results=[
            _check(2.0, success="FAIL", property="energy", metric="rmse"),
            _check(3.0, success="FAIL", property="forces", metric="mae"),
        ]
    )
    loop = SimpleNamespace(result=SimpleNamespace(iterations=[prev_iter]))
    state = SimpleNamespace(
        task_results=SimpleNamespace(validity_get_results=[]),
        accuracy_checks_results=[_check(2.5, success="FAIL", property="forces", metric="mae")],
        al_finished_reason=None,
    )

    cb.after_convergence(loop, state)

    assert cb._no_improve_steps == 0
    assert cb._info_history == []


def test_early_stop_sets_finished_reason_after_patience():
    cb = EarlyStopAccuracy(patience=1)
    cb._no_improve_steps = 1

    prev_iter = SimpleNamespace(accuracy_checks_results=[_check(1.0, success="FAIL")])
    loop = SimpleNamespace(result=SimpleNamespace(iterations=[prev_iter]))
    state = SimpleNamespace(
        task_results=SimpleNamespace(validity_get_results=[]),
        accuracy_checks_results=[_check(2.0, success="FAIL")],
        al_finished_reason=None,
        skip={},
    )

    cb.after_convergence(loop, state)

    assert state.al_finished_reason is not None
    assert "EarlyStopAccuracy: accuracy errors not reduced for 1 iterations." in state.al_finished_reason
    assert "Errors| energy-rmse: 2.000 (>= prev 1)" in state.al_finished_reason
    assert state.skip[IterPhase.SPLIT_AND_ADD] == "EarlyStopAccuracy"
    assert state.skip[IterPhase.TRAIN] == "EarlyStopAccuracy"


def test_early_stop_uses_latest_iteration_as_reference_after_reset():
    cb = EarlyStopAccuracy(patience=2)
    cb._no_improve_steps = 0

    first_iter = SimpleNamespace(accuracy_checks_results=[_check(100.0, success="FAIL")])
    last_iter = SimpleNamespace(accuracy_checks_results=[_check(1.0, success="FAIL")])
    loop = SimpleNamespace(result=SimpleNamespace(iterations=[first_iter, last_iter]))
    state = SimpleNamespace(
        task_results=SimpleNamespace(validity_get_results=[]),
        accuracy_checks_results=[_check(2.0, success="FAIL")],
        al_finished_reason=None,
        skip={},
    )

    cb.after_convergence(loop, state)

    assert cb._no_improve_steps == 1
    assert cb._info_history == ["Errors| energy-rmse: 2.000 (>= prev 1)"]


def test_early_stop_validity_skips_without_task_results():
    cb = EarlyStopValidity()
    state = SimpleNamespace(task_results=None, al_finished_reason=None, skip={})

    cb.after_run_task(SimpleNamespace(), state)

    assert state.al_finished_reason is None
    assert state.skip == {}


def test_early_stop_validity_ignores_non_selected_failures():
    cb = EarlyStopValidity(check_results_query="property='trajectory',metric='reasonable_final_frame'")
    state = SimpleNamespace(
        task_results=_task_results(
            _check(1.0, success="FAIL", property="energy", metric="rmse"),
            _check(0.0, success="OK", property="trajectory", metric="reasonable_final_frame"),
        ),
        al_finished_reason=None,
        skip={},
    )

    cb.after_run_task(SimpleNamespace(), state)

    assert state.al_finished_reason is None
    assert state.skip == {}


def test_early_stop_validity_stops_preemptively_on_selected_failure():
    cb = EarlyStopValidity(check_results_query="property='trajectory',metric='reasonable_final_frame'")
    state = SimpleNamespace(
        task_results=_task_results(
            _check(
                "bad geometry",
                success="FAIL",
                property="trajectory",
                metric="reasonable_final_frame",
                value_type="str",
            )
        ),
        al_finished_reason=None,
        skip={},
    )

    cb.after_run_task(SimpleNamespace(), state)

    assert state.al_finished_reason is not None
    assert "EarlyStopValidity: selected validity checks failed." in state.al_finished_reason
    assert "trajectory-reasonable_final_frame: bad geometry" in state.al_finished_reason
    assert state.skip[IterPhase.LABEL] == "EarlyStopValidity"
    assert state.skip[IterPhase.ACCURACY] == "EarlyStopValidity"
    assert state.skip[IterPhase.CONVERGENCE] == "EarlyStopValidity"
    assert state.skip[IterPhase.SPLIT_AND_ADD] == "EarlyStopValidity"
    assert state.skip[IterPhase.TRAIN] == "EarlyStopValidity"
