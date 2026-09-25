from __future__ import annotations

from types import SimpleNamespace

import pytest

from scm.active_learning.callbacks import StopBelowActiveSystems
from scm.active_learning.loop.iteration_phase import IterPhase
from scm.active_learning.loop.iteration_state import IterationState


def _loop_with_active(n_active: int) -> SimpleNamespace:
    return SimpleNamespace(journey=SimpleNamespace(type="MoleculesJourney", n_active_molecules=lambda: n_active))


def test_stops_and_skips_all_phases_below_min_active():
    state = IterationState(start_engine=None, iteration_al=3)

    StopBelowActiveSystems(min_active=2).after_convergence(_loop_with_active(1), state)

    assert "1 active system(s) left" in state.al_finished_reason
    assert set(state.skip) == set(IterPhase)
    assert IterPhase.TRAIN in state.skip


@pytest.mark.parametrize("n_active", [2, 5])
def test_keeps_running_at_or_above_min_active(n_active):
    state = IterationState(start_engine=None, iteration_al=3)

    StopBelowActiveSystems(min_active=2).after_convergence(_loop_with_active(n_active), state)

    assert state.al_finished_reason is None
    assert state.skip == {}


def test_requires_a_journey_counting_active_systems():
    state = IterationState(start_engine=None, iteration_al=0)
    al_loop = SimpleNamespace(journey=SimpleNamespace(type="SimpleActiveLearningJourney"))

    with pytest.raises(TypeError):
        StopBelowActiveSystems().after_convergence(al_loop, state)


def test_is_parsed_as_a_loop_callback():
    from pydantic import TypeAdapter

    from scm.active_learning.callbacks import UnionALCallbacks

    callback = TypeAdapter(UnionALCallbacks).validate_python({"type": "StopBelowActiveSystems", "min_active": 2})

    assert isinstance(callback, StopBelowActiveSystems)
