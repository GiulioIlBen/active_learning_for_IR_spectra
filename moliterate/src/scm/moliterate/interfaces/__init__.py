from typing import Union

from .ase_database import ASEMolData
from .in_memory import InMemoryMolData
from .params_data import ParAMSData
from .rkf_files import RKFMolData

UnionInterfaces = Union[RKFMolData, ParAMSData, InMemoryMolData, ASEMolData]

UnionInterfaceWriters = Union[InMemoryMolData, ASEMolData]

__all__ = ["RKFMolData", "ParAMSData", "InMemoryMolData", "ASEMolData", "UnionInterfaces", "UnionInterfaceWriters"]
