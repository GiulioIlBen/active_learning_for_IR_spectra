from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from .ams import AMSEngine
from .ase_engine import ASEEngine
from .atomic_container import AtomisticContainer, PLAMSMolecule, SCMChemicalSystem
from .core import Engine
from .mlip_ams import ParAMSEngine
from .torchsim import TorchEngine

ConcreteEngines = Annotated[
    Union[AMSEngine, ParAMSEngine, ASEEngine, TorchEngine],
    Field(discriminator="type"),
]

ConcreteSystems = Annotated[Union[PLAMSMolecule, SCMChemicalSystem], Field(discriminator="type")]


def to_ams_engine(eng: Union[AMSEngine, ParAMSEngine, ASEEngine]) -> AMSEngine:
    if isinstance(eng, ParAMSEngine):
        return eng.ams_engine
    if isinstance(eng, AMSEngine):
        return eng
    if isinstance(eng, ASEEngine):
        return eng.ams_engine
    raise TypeError(f"Invalid engine type: {type(eng)=}")


__all__ = [
    "SCMChemicalSystem",
    "AtomisticContainer",
    "PLAMSMolecule",
    "Engine",
    "AMSEngine",
    "ASEEngine",
    "TorchEngine",
    "ParAMSEngine",
    "ConcreteEngines",
    "to_ams_engine",
    "ConcreteSystems",
]
