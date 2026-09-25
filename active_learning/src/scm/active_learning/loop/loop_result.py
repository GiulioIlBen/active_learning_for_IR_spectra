from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel

from scm.active_learning.engines import ConcreteEngines
from scm.active_learning.loop.iteration_state import IterationState

TableKind = Literal["validity", "accuracy", "getters", "summary", "timings", "train", "train_check", "summary_validity"]


class LoopResult(BaseModel):
    iterations: List[IterationState] = []
    final_engine: Optional[ConcreteEngines] = None
    exit_message: Optional[str] = None

    def collect_table(self, options: TableKind = "summary", **kwargs):
        iter_options: Dict[str, Callable[[IterationState], Iterable[Dict]]] = {
            "validity": lambda state, **kwargs: state.iter_dump_validities(),
            "accuracy": lambda state, **kwargs: state.iter_dump_accuracies(),
            "getters": lambda state, **kwargs: state.iter_dump_getters(),
            "train": lambda state, **kwargs: state.iter_dump_train(),
            "summary": lambda state, **kwargs: [state.summary()],
            "summary_validity": lambda state, **kwargs: state.iter_dump_validities_summary(**kwargs),
            "timings": lambda state, **kwargs: [state.dump_timings(**kwargs)],
            "train_check": lambda state, **kwargs: state.iter_dump_train_check(**kwargs),
        }
        for state in self.iterations:
            for x in iter_options[options](state, **kwargs):
                y = state.dump_iteration_al()
                y.update(x)
                yield y
