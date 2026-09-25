from functools import cached_property
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
from pydantic import PrivateAttr, field_validator

from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.transforms.core import BaseEntryTransform
from scm.moliterate.utils.unit_conversion import conversion_ratio


class MinMaxAtomsDistance(BaseEntryTransform):
    type: Literal["MinMaxAtomsDistance"] = "MinMaxAtomsDistance"  # Field discriminator
    min_out_key: Optional[str] = "min_distance"
    max_out_key: Optional[str] = "max_distance"
    mic: bool = False

    def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return properties_info

    def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        updated = list(properties_info or [])
        for key in (self.min_out_key, self.max_out_key):
            if key is None or any(p.name == key for p in updated):
                continue
            updated.append(PropertyInfo(name=key, shape="float"))
        return updated

    def __call__(self, data: ChemDataEntry) -> ChemDataEntry:
        distances = data.atoms.get_all_distances(mic=self.mic, vector=False)
        top_triangle = np.abs(distances[np.triu_indices(distances.shape[0], k=1)])
        if self.min_out_key is not None:
            data.properties[self.min_out_key] = float(top_triangle.min())
        if self.max_out_key is not None:
            data.properties[self.max_out_key] = float(top_triangle.max())
        return data


class PropertyTransform(BaseEntryTransform):
    type: Literal["PropertyTransform"] = "PropertyTransform"  # Field discriminator
    property_key: str = "energy"
    post_process: Optional[Literal["l2_max", "l2", "max"]] = None
    unit_transform: Optional[Tuple[str, str]] = None
    property_key_out: str = "__prop_trn_out__"

    def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return [x for x in properties_info if x.name == self.property_key]

    def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return [*properties_info, PropertyInfo(name=self.property_key_out)]

    @field_validator("unit_transform")
    @classmethod
    def _check_unit_transform(cls, value):
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"Must be 2 but found len(unit_transform)={len(value)}")
        return tuple(value)

    def __call__(self, row: ChemDataEntry) -> ChemDataEntry:
        value = row.properties[self.property_key]
        if self.post_process == "l2_max":
            value = (value**2).sum(1).max() ** 0.5
        elif self.post_process == "l2":
            value = (value**2).sum(1) ** 0.5
        elif self.post_process == "max":
            value = value.max()
        elif self.post_process is not None:
            raise ValueError(f"Not allowed value: {self.post_process}")
        if self.unit_transform is not None:
            value *= conversion_ratio(*self.unit_transform)
        row.properties[self.property_key_out] = value
        return row


class FmaxTransform(BaseEntryTransform):
    type: Literal["FmaxTransform"] = "FmaxTransform"
    forces_keys: Tuple[str, ...] = ("Gradients", "forces", "EngineGradients")
    prepend_keys: Tuple[str, ...] = ("", "History%", "MDHistory%")
    property_key_out: str = "fmax"

    def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return self.composed_transform.properties_in(properties_info)

    def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return self.composed_transform.properties_out(properties_info)

    @cached_property
    def composed_transform(self):
        from scm.moliterate.transforms import (
            ConvertNamesTransform,
            ReshapeTransform,
            ScalePropTransform,
            ToArrayTransform,
        )
        from scm.moliterate.transforms.composed_transforms import ComposedTransform

        names = [a + k for k in self.forces_keys for a in self.prepend_keys]
        trn = ComposedTransform(
            transforms=[
                ToArrayTransform(prop_names=names),
                *[ScalePropTransform(key_in=k, coeff_scale=-1) for k in names if "gradients" in k.lower()],
                ConvertNamesTransform(
                    prop_names_convert={a + k: "forces" for k in self.forces_keys for a in self.prepend_keys}
                ),
                ReshapeTransform(props_shapes={"forces": [-1, 3]}),
                PropertyTransform(property_key="forces", post_process="l2_max", property_key_out=self.property_key_out),
            ]
        )
        return trn

    def __call__(self, row: ChemDataEntry) -> ChemDataEntry:
        return self.composed_transform(row)

    @property
    def name(self):
        return "fmax"
