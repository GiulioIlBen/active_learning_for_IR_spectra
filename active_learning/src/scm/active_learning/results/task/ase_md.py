from __future__ import annotations

from typing import List, Literal

from pydantic import Field

from scm.active_learning.engines import ConcreteEngines
from scm.active_learning.tasks import ASEMolecularDynamics

from .core import TaskResult


class ASEMolecularDynamicsResult(TaskResult):
    type: Literal["ASEMolecularDynamicsResult"] = "ASEMolecularDynamicsResult"
    task: ASEMolecularDynamics
    engine: ConcreteEngines
    save_folder: str
    trajectory_path: str
    restart_used: bool = False
    stop_triggered: bool = False
    stop_reasons: List[str] = Field(default_factory=list)
    requested_steps: int = 0
    executed_steps: int = 0
    written_frames: int = 0
    dynamics_type: str = ""

    @property
    def from_task(self):
        return self.task

    @property
    def from_engine(self):
        return self.engine
