from __future__ import annotations

from typing import Literal

from scm.active_learning.engines import ConcreteEngines
from scm.active_learning.tasks.torchsim_md import TorchSimMDTask

from .core import TaskResult


class TorchSimMDResult(TaskResult):
    type: Literal["TorchSimMDResult"] = "TorchSimMDResult"
    task: TorchSimMDTask
    engine: ConcreteEngines
    save_folder: str
    trajectory_path: str
    requested_steps: int = 0
    executed_steps: int = 0
    integrator: str = ""
    final_energy: float = 0.0

    @property
    def from_task(self):
        return self.task

    @property
    def from_engine(self):
        return self.engine
