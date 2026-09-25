from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from scm.moliterate.transforms.atoms_stats import MinMaxAtomsDistance, PropertyTransform
from scm.moliterate.transforms.property_info_transform import (
    ConvertNamesTransform,
    MetadataAddTransform,
    ReshapeTransform,
    ScalePropTransform,
    ToArrayTransform,
)

UnionRowTransform = Annotated[
    Union[
        PropertyTransform,
        MinMaxAtomsDistance,
        ToArrayTransform,
        ConvertNamesTransform,
        ReshapeTransform,
        ScalePropTransform,
        MetadataAddTransform,
    ],
    Field(discriminator="type"),
]


__all__ = [
    "PropertyTransform",
    "MinMaxAtomsDistance",
    "ToArrayTransform",
    "ConvertNamesTransform",
    "ReshapeTransform",
    "ScalePropTransform",
    "MetadataAddTransform",
    "UnionRowTransform",
]
