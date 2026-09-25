from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from scm.moliterate.core import (
    ChemDataEntry,
    ChemDataSetFormat,
    ConcatChemDataSet,
    PropertyInfo,
    create_dataset,
    load_dataset,
)
from scm.moliterate.interfaces import UnionInterfaces, UnionInterfaceWriters

ConcreteInterfacesWriters = Annotated[UnionInterfaceWriters, Field(discriminator="type")]

ConcreteInterfaces = Annotated[Union[UnionInterfaces, ConcatChemDataSet], Field(discriminator="type")]

ConcatChemDataSet.model_rebuild()

__all__ = [
    "ChemDataEntry",
    "create_dataset",
    "load_dataset",
    "ConcatChemDataSet",
    "PropertyInfo",
    "ChemDataSetFormat",
    "ConcreteInterfaces",
    "ConcreteInterfacesWriters",
]
