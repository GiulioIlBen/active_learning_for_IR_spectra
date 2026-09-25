from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from .ams_conformers import AMSConformersResults
from .ams_task import AMSTaskResult
from .ase_md import ASEMolecularDynamicsResult
from .core import TaskResult
from .loader import load_task_result
from .sp_labeller import SPLabellingResults
from .torchsim_go import TorchSimGOResult
from .torchsim_md import TorchSimMDResult

ConcreteTaskResult = Annotated[
    Union[
        SPLabellingResults,
        AMSTaskResult,
        AMSConformersResults,
        ASEMolecularDynamicsResult,
        TorchSimMDResult,
        TorchSimGOResult,
    ],
    Field(discriminator="type"),
]


__all__ = [
    "ConcreteTaskResult",
    "TaskResult",
    "SPLabellingResults",
    "AMSConformersResults",
    "AMSTaskResult",
    "ASEMolecularDynamicsResult",
    "TorchSimMDResult",
    "TorchSimGOResult",
    "load_task_result",
]
