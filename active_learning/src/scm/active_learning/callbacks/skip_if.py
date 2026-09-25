from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal

from scm.active_learning.callbacks.core import ALCallback
from scm.active_learning.logging import log_level
from scm.active_learning.loop.iteration_phase import IterPhase
from scm.active_learning.results import JourneyAdvanceResult

if TYPE_CHECKING:
    from scm.active_learning.loop.iteration_state import IterationState
    from scm.active_learning.loop.loop import ActiveLearningLoop


class SkipIf(ALCallback):
    type: Literal["SkipIf"] = "SkipIf"
    if_start_engine_is_not_finetunable: List[IterPhase] = [
        IterPhase.ACCURACY,
        IterPhase.CONVERGENCE,
    ]
    when_true_restart_journey: bool = True
    default_accuracy_checks_results: List = []
    default_journey_advance_results: JourneyAdvanceResult = JourneyAdvanceResult()
    message: str = "SkipIf: start engine is not finetunable"

    def start_iter_loop(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        to_skip = {}
        for phase in self.if_start_engine_is_not_finetunable:
            if state.start_engine is not None and not state.start_engine.is_finetunable():
                to_skip[phase] = self.message
        if IterPhase.ACCURACY in to_skip and state.accuracy_checks_results is None:
            state.accuracy_checks_results = self.default_accuracy_checks_results
        if IterPhase.CONVERGENCE in to_skip and state.journey_advance_results is None:
            state.journey_advance_results = self.default_journey_advance_results
        state.skip.update(to_skip)

    def after_train(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        skip_on = all(
            [x in state.skip and state.skip.get(x) == self.message for x in self.if_start_engine_is_not_finetunable]
        )
        if skip_on and self.when_true_restart_journey:
            al_loop.journey.restart_journey()
            log_level("SkipIf: Restarted Journey", level="SUCCESS")
