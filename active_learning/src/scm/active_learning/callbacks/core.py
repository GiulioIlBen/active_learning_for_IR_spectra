from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from scm.active_learning.loop.iteration_state import IterationState
    from scm.active_learning.loop.loop import ActiveLearningLoop


class ALCallback(BaseModel):
    # type: str  # It must have a type, and it is used as discriminator

    def before_loop(self, al_loop: "ActiveLearningLoop", state=None) -> None:
        pass

    def after_loop(self, al_loop: "ActiveLearningLoop", state=None) -> None:
        pass

    def start_iter_loop(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        pass

    def after_run_task(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        pass

    def after_label(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        pass

    def after_accuracy(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        pass

    def after_convergence(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        pass

    def after_split_and_add(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        pass

    def after_train(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        pass

    def after_train_check(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
