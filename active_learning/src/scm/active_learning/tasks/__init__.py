from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from .ams_conformers_task import AMSConformersTask
from .ams_task import AMSGOIRTask, AMSMDTask, AMSTask
from .ase_md import ASEMolecularDynamics
from .core import Task
from .sp_labeller import SPLabeller, SPLabellerData
from .torchsim_go import TorchSimGOTask
from .torchsim_md import TorchSimMDTask

ConcreteTask = Annotated[
    Union[
        SPLabellerData,
        AMSTask,
        AMSMDTask,
        AMSGOIRTask,
        AMSConformersTask,
        ASEMolecularDynamics,
        TorchSimMDTask,
        TorchSimGOTask,
    ],
    Field(discriminator="type"),
]


__all__ = [
    "ConcreteTask",
    "Task",
    "SPLabeller",
    "SPLabellerData",
    "ASEMolecularDynamics",
    "AMSMDTask",
    "AMSGOIRTask",
    "AMSTask",
    "AMSConformersTask",
    "TorchSimMDTask",
    "TorchSimGOTask",
]
