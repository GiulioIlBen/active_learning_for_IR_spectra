from typing import Dict

from scm.active_learning.journey_scheduler.molecule_journey import MoleculeJourneyRegistry


def _make_registry(n_molecules: int, tasks: Dict[str, int]) -> MoleculeJourneyRegistry:
    registry = MoleculeJourneyRegistry()
    registry.clear_init_journey(n_molecules=n_molecules, tasks=tasks)
    return registry


def test_registry_init_and_queries():
    registry = _make_registry(3, {"t1": 1, "t2": 2})

    assert len(registry.molecules) == 3
    assert [m.status for m in registry.molecules] == ["Pending", "Pending", "Pending"]
    assert registry.n_pending_molecules() == 3
    assert registry.n_running() == 0
    assert registry.n_inactivated_molecules() == 0
    assert [idx for idx, _ in registry.molecules_query("All")] == [0, 1, 2]
    assert list(registry.molecules_query("ExcludePending")) == []


def test_registry_activate_and_current_running():
    registry = _make_registry(2, {"t1": 0, "t2": 0})

    registry.activate_n_running(batch=1, batch_by="molecules")
    assert registry.n_running() == 1
    assert set(registry.current_running()) == {(0, "t1"), (0, "t2")}

    registry.molecules[0].tasks["t1"].status = "Finished"
    assert list(registry.current_running()) == [(0, "t2")]


def test_registry_register_attempt_tracks_steps_record():
    registry = _make_registry(3, {"t1": 0})

    registry.activate_n_running(batch=2, batch_by="molecules")
    registry.register_attempt()

    assert registry.steps_record == [[0, 1]]
    assert [m.attempt for m in registry.molecules] == [1, 1, 0]
    assert [m.tasks["t1"].attempt for m in registry.molecules] == [1, 1, 0]


def test_registry_success_and_failure_transitions():
    registry = _make_registry(1, {"t1": 0})
    registry.activate_n_running(batch=1, batch_by="molecules")
    registry.register_attempt()
    registry.set_molecule_succeeded(0)
    assert registry.molecules[0].status == "Succeeded"
    assert registry.molecules[0].tasks["t1"].status == "Succeeded"

    registry = _make_registry(1, {"t1": 0})
    registry.activate_n_running(batch=1, batch_by="molecules")
    registry.register_attempt()
    finished = registry.record_molecule_failure_on_task(0, "t1", "fail")
    assert finished is True
    assert registry.molecules[0].status == "Finished"
    assert registry.molecules[0].tasks["t1"].status == "Finished"
    assert registry.molecules[0].tasks["t1"].attempt_messages == {1: "fail"}
