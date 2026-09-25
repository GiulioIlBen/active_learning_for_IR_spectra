from __future__ import annotations

from types import SimpleNamespace

from scm.active_learning.callbacks.manual_stopper import ManualStopper
from scm.active_learning.loop.iteration_phase import IterPhase


def test_manual_stopper_ignores_missing_file(tmp_path):
    state = SimpleNamespace(al_finished_reason=None, skip={})
    callback = ManualStopper(folder_path=tmp_path, file_name="missing.stop")
    loop = SimpleNamespace(query_callbacks=lambda _: [])

    callback.start_iter_loop(al_loop=loop, state=state)

    assert state.al_finished_reason is None
    assert state.skip == {}


def test_manual_stopper_ignores_empty_file(tmp_path):
    stop_file = tmp_path / "manual.stop"
    stop_file.write_text("   \n", encoding="utf-8")
    state = SimpleNamespace(al_finished_reason=None, skip={})
    callback = ManualStopper(folder_path=tmp_path, file_name=stop_file.name)
    loop = SimpleNamespace(query_callbacks=lambda _: [])

    callback.start_iter_loop(al_loop=loop, state=state)

    assert state.al_finished_reason is None
    assert state.skip == {}


def test_manual_stopper_stops_loop_when_message_exists(tmp_path):
    stop_file = tmp_path / "manual.stop"
    stop_file.write_text("please stop after this iteration", encoding="utf-8")
    state = SimpleNamespace(al_finished_reason=None, skip={})
    callback = ManualStopper(folder_path=tmp_path, file_name=stop_file.name)
    loop = SimpleNamespace(query_callbacks=lambda _: [])

    callback.start_iter_loop(al_loop=loop, state=state)

    assert state.al_finished_reason == "ManualStopper: stop requested. please stop after this iteration"
    assert state.skip == {phase: "ManualStopper" for phase in IterPhase}


def test_manual_stopper_uses_loop_folder_when_folder_path_is_none(tmp_path):
    stop_file = tmp_path / "STOP_ACTIVE_LEARNING"
    stop_file.write_text("stop from loop folder", encoding="utf-8")
    state = SimpleNamespace(al_finished_reason=None, skip={})
    folder_callback = SimpleNamespace(run_dir=lambda: tmp_path)
    callback = ManualStopper()
    loop = SimpleNamespace(query_callbacks=lambda _: [folder_callback])

    callback.start_iter_loop(al_loop=loop, state=state)

    assert state.al_finished_reason == "ManualStopper: stop requested. stop from loop folder"
