from __future__ import annotations

from typing import Literal

from scm.active_learning.callbacks.core import ALCallback
from scm.active_learning.loop.iteration_phase import IterPhase


class NoPostFilterData(ALCallback):
    type: Literal["NoPostFilterData"] = "NoPostFilterData"
    skip_training_too: bool = False

    def after_label(self, al_loop, state) -> None:
        if state.ds_post_filters is None:
            return
        message = "Post-filter dataset is empty"
        if len(state.ds_post_filters) == 0:
            state.skip[IterPhase.ACCURACY] = message
            state.skip[IterPhase.TRAIN_CHECK] = message
            state.accuracy_checks_results = []
            state.skip[IterPhase.SPLIT_AND_ADD] = message
            if self.skip_training_too:
                state.skip[IterPhase.TRAIN] = message
