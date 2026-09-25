from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from tabulate import tabulate

from scm.active_learning.callbacks.core import ALCallback
from scm.active_learning.logging import log_level

if TYPE_CHECKING:
    from scm.active_learning.loop.iteration_state import IterationState
    from scm.active_learning.loop.loop import ActiveLearningLoop


class ALMinimalLogger(ALCallback):
    type: Literal["ALMinimalLogger"] = "ALMinimalLogger"
    log_journey_table: bool = True
    log_al_iter_index: bool = True
    log_history_table: bool = True
    log_final_summary: bool = True

    def before_loop(self, al_loop, state=None) -> None:
        if not self.log_journey_table:
            return
        log_level(message="\n{journey_table}", level="INFO", journey_table=al_loop.journey.journey_table())

    def start_iter_loop(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        if not self.log_al_iter_index:
            return
        log_level(
            message="===================== iAL:{iAL:02} =====================",
            iAL=state.iteration_al,
            level="INFO",
        )

    def after_train(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        if not self.log_history_table:
            return
        log_level(
            message="\n{history_table}",
            level="INFO",
            history_table=al_loop.journey.history_table(),
        )

    def after_loop(self, al_loop: "ActiveLearningLoop", state=None) -> None:
        if not self.log_final_summary:
            return
        log_level(
            message="ActiveLearningLoop summary:\n{final_summary}",
            level="INFO",
            final_summary=tabulate(
                {k: [v] for k, v in al_loop.analysis.get_summary().items()},
                headers="keys",
            ),
        )
