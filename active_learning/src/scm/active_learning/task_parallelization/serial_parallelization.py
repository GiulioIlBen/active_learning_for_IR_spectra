from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Tuple

from .core import ParallelStrategy

if TYPE_CHECKING:
    from scm.active_learning.engines import ConcreteEngines
    from scm.active_learning.tasks import ConcreteTask


class SerialStrategy(ParallelStrategy):
    type: Literal["SerialStrategy"] = "SerialStrategy"  # It must have a type, and it is used as discriminator

    def run_tasks(self, task_engine_coll: List[Tuple["ConcreteTask", "ConcreteEngines"]]):
        return [task_i.run(engine=engine_i) for task_i, engine_i in task_engine_coll]
