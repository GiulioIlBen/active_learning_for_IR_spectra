from __future__ import annotations

from typing import Literal

from scm.active_learning.engines import ConcreteEngines
from scm.active_learning.tasks.torchsim_go import TorchSimGOTask

from .core import TaskResult


class TorchSimGOResult(TaskResult):
    type: Literal["TorchSimGOResult"] = "TorchSimGOResult"
    task: TorchSimGOTask
    engine: ConcreteEngines
    save_folder: str
    trajectory_path: str
    optimizer: str = ""
    convergence_kind: Literal["energy", "force"] = "force"
    final_energy: float = 0.0

    @property
    def from_task(self):
        return self.task

    @property
    def from_engine(self):
        return self.engine
