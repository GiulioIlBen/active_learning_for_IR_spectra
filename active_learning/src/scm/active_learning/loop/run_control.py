from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from scm.active_learning.engines import ConcreteEngines
    from scm.active_learning.loop.loop import ActiveLearningLoop


class ActiveLearningRunControl(BaseModel):
    mode: Literal["fresh", "resume"] = "fresh"
    cleanup_resume_iteration: bool = True

    @property
    def is_resume(self) -> bool:
        return self.mode == "resume"

    def start_iteration(self, loop: "ActiveLearningLoop") -> int:
        if not self.is_resume:
            return loop.iterable_loop.start

        if loop.current_state.iteration_al >= 0:
            return loop.current_state.iteration_al
        return len(loop.result.iterations)

    def start_engine(self, loop: "ActiveLearningLoop") -> "ConcreteEngines":
        if not self.is_resume:
            if loop.start_engine is None:
                raise ValueError("Cannot start active learning: start_engine is missing.")
            return loop.start_engine

        if loop.current_state.start_engine is not None:
            return loop.current_state.start_engine
        if loop.result.final_engine is not None:
            return loop.result.final_engine
        if loop.result.iterations:
            train_results = loop.result.iterations[-1].train_results
            if train_results is not None:
                return train_results.engine
        raise ValueError("Cannot resume active learning: no continuation engine found.")

    def should_reset_shared_dataset(self) -> bool:
        return not self.is_resume

    def should_import_start_engine_data(self) -> bool:
        return not self.is_resume

    def should_cleanup_resume_iteration(self) -> bool:
        return self.is_resume and self.cleanup_resume_iteration
