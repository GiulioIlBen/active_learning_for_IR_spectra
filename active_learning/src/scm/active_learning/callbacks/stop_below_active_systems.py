from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from scm.active_learning.callbacks.core import ALCallback
from scm.active_learning.logging import log_level
from scm.active_learning.loop.iteration_phase import IterPhase

if TYPE_CHECKING:
    from scm.active_learning.loop.iteration_state import IterationState
    from scm.active_learning.loop.loop import ActiveLearningLoop


class StopBelowActiveSystems(ALCallback):
    """Stop the loop as soon as convergence leaves fewer than ``min_active`` active systems.

    The check runs right after the convergence phase, so the current iteration is not split nor trained: the final
    engine is the one trained in the previous iteration and the frames labelled in this iteration are not used.
    """

    type: Literal["StopBelowActiveSystems"] = "StopBelowActiveSystems"
    min_active: int = Field(default=2, gt=0)
    skip_message: str = "StopBelowActiveSystems"

    def after_convergence(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        n_active_molecules = getattr(al_loop.journey, "n_active_molecules", None)
        if n_active_molecules is None:
            raise TypeError(f"StopBelowActiveSystems needs a journey with n_active_molecules(): {al_loop.journey.type}")
        n_active = n_active_molecules()
        if n_active >= self.min_active:
            return

        state.al_finished_reason = (
            f"StopBelowActiveSystems: {n_active} active system(s) left, fewer than min_active={self.min_active}"
        )
        state.skip.update({phase: self.skip_message for phase in IterPhase})
        log_level("{reason}", level="SUCCESS", reason=state.al_finished_reason)
