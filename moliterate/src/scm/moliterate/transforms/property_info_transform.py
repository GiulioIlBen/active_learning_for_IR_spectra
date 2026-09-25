from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import ConfigDict, PrivateAttr

from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.transforms.core import BaseEntryTransform

__all__ = [
    "ToArrayTransform",
    "ReshapeTransform",
    "ConvertNamesTransform",
    "ScalePropTransform",
    "MetadataAddTransform",
]


class ToArrayTransform(BaseEntryTransform):
    type: Literal["ToArrayTransform"] = "ToArrayTransform"  # Field discriminator
    prop_names: List[str]
    array_type: Literal["np"] = "np"
    _converter: Callable[[Any], Any] = PrivateAttr()

    def model_post_init(self, __context) -> None:
        converters = {"np": np.asarray}
        self._converter = converters[self.array_type]

    def to_array(self, val):
        return self._converter(val)

    def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return [x for x in properties_info if x.name in self.prop_names]

    def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return properties_info

    def __call__(self, data: ChemDataEntry) -> ChemDataEntry:
        props = data.properties
        for prop_name in self.prop_names:
            if prop_name in props:
                props[prop_name] = self.to_array(props[prop_name])
        return data


class ReshapeTransform(BaseEntryTransform):
    type: Literal["ReshapeTransform"] = "ReshapeTransform"  # Field discriminator
    props_shapes: Dict[str, List[Union[int, Literal["nAtoms"]]]]
    array_type: Literal["np"] = "np"
    _reshape: Callable[[Any, Tuple[int, ...]], Any] = PrivateAttr()

    def model_post_init(self, __context) -> None:
        converters = {"np": np.reshape}
        self._reshape = converters[self.array_type]

    def reshape_arr(self, val, shape):
        return self._reshape(val, tuple(shape))

    def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return [x for x in properties_info if x.name in self.props_shapes]

    def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        if not properties_info:
            return properties_info

        updated: List[PropertyInfo] = []
        for prop_info in properties_info:
            shape = self.props_shapes.get(prop_info.name)
            if shape is None:
                updated.append(prop_info)
                continue
            normalized_shape: Union[Tuple[Union[int, Literal["nAtoms"]], ...], Literal["float"]]
            if isinstance(shape, str) and shape == "float":
                normalized_shape = "float"
            else:
                normalized_shape = tuple(shape)
            if prop_info.shape == normalized_shape:
                updated.append(prop_info)
            else:
                updated.append(prop_info.model_copy(update={"shape": normalized_shape}))
        return updated

    def __call__(self, data: ChemDataEntry) -> ChemDataEntry:
        props = data.properties
        for prop_name, prop_shape in self.props_shapes.items():
            if prop_name in props and prop_shape is not None:
                try:
                    if "nAtoms" in prop_shape:
                        prop_shape = self.convert_special_shape(prop_shape, len(data.atoms))
                    arr = props[prop_name]
                    props[prop_name] = self.reshape_arr(arr, prop_shape)
                except ValueError as e:
                    raise ValueError(f"at {prop_name=} {prop_shape=} with {arr=} \n Exception raised: {e}")
        return data

    @staticmethod
    def convert_special_shape(
        shape: Union[List[Union[int, Literal["nAtoms"]]], Literal["float"]], n_atoms: int, shape_ref=None
    ) -> Union[Tuple[int, ...], Literal["float"]]:
        if isinstance(shape, str) and shape == "float":
            return shape
        vals = list(n_atoms if item == "nAtoms" else item for item in shape)
        if -1 in vals and shape_ref is not None:
            idx_min_one = vals.index(-1)
            vals[idx_min_one] = shape_ref[idx_min_one]
        return tuple(vals)


class ConvertNamesTransform(BaseEntryTransform):
    prop_names_convert: Dict[str, str]
    options: Literal["override"] = "override"
    type: Literal["ConvertNamesTransform"] = "ConvertNamesTransform"  # Field discriminator

    def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return [x for x in properties_info if x.name in self.prop_names_convert]

    def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        updated: List[PropertyInfo] = []
        if self.options == "override":
            for prop_info in properties_info:
                target_name = self.prop_names_convert.get(prop_info.name, prop_info.name)
                updated.append(prop_info.model_copy(update={"name": target_name}))
        return updated

    def __call__(self, data: ChemDataEntry) -> ChemDataEntry:
        props = data.properties
        metadata = data.metadata
        if self.options == "override":
            for prop_name, out_name in self.prop_names_convert.items():
                if prop_name in props:
                    props[out_name] = props.pop(prop_name)
                if prop_name in metadata:
                    metadata[out_name] = metadata.pop(prop_name)
        return data


class ScalePropTransform(BaseEntryTransform):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["ScalePropTransform"] = "ScalePropTransform"  # Field discriminator
    key_in: str
    coeff_scale: float
    key_out: Optional[str] = None

    def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return [x for x in properties_info if x.name == self.key_in]

    def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        if self.key_out is None:
            return properties_info
        return ConvertNamesTransform(prop_names_convert={self.key_in: self.key_out}).properties_out(properties_info)

    def __call__(self, data: ChemDataEntry) -> ChemDataEntry:
        if self.key_in in data.properties:
            key_o = self.key_out or self.key_in
            data.properties[key_o] = data.properties.pop(self.key_in) * self.coeff_scale
        return data


class MetadataAddTransform(BaseEntryTransform):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["MetadataAddTransform"] = "MetadataAddTransform"  # Field discriminator

    key_values: Dict[str, Union[str, int, float]]
    overwrite: bool = True

    def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return properties_info

    def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        return properties_info

    def __call__(self, data: ChemDataEntry) -> ChemDataEntry:
        if self.overwrite:
            data.metadata.update(self.key_values)
            return data
        md = self.key_values.copy()
        md.update(data.metadata)
        data.metadata = md
        return data


##
