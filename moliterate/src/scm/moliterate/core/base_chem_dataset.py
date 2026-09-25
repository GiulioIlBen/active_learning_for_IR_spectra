import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, TypeVar, Union, overload

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.transforms import UnionRowTransform
from scm.moliterate.transforms.composed_transforms import ComposedTransform


class IterDataCoreError(Exception):
    pass


class IterDataKeyError(KeyError):
    pass


TBaseChemDataSet = TypeVar("TBaseChemDataSet", bound="BaseChemDataSet")


class BaseChemDataSet(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # type: Literal[""] = ""  # used as discriminator

    # if it is connected to a file please use data_source
    # data_source:
    absolute_idxs: Optional[np.ndarray] = None
    transforms: List[UnionRowTransform] = []

    @field_validator("absolute_idxs", mode="before")
    @classmethod
    def _absolute_idxs_to_array(cls, value):
        if value is None:
            return None
        arr = np.asarray(value, dtype=cls._dtype_absolute_idx())
        return arr

    @field_serializer("absolute_idxs", when_used="json")
    def _absolute_idxs_serializer(self, value):
        if value is None:
            return None
        serialized = value.tolist()
        return serialized

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        trasf = str([x.__class__ for x in self.transforms]) if self.transforms is not None else 0
        return (
            f"{self.__class__.__name__}(len={len(self)}, "
            f"properties={[x.name for x in self.available_properties]}, "
            f"transforms={trasf} at {hex(id(self))})"
        )

    ###############################################################################################
    ###################               abstract/to-override methods              ###################
    ###############################################################################################

    @abstractmethod
    def total_len(self) -> int:
        """method to get the total len of the data"""

    @property
    @abstractmethod
    def available_properties(self) -> List[PropertyInfo]:
        """Available properties in the dataset"""

    @property
    def distance_unit(self) -> str:
        return "Ang"

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Global metadata"""

    # TBN: to make more pythonic the BaseChemDataSet
    # while keeping the formalism of BaseModel (gaining validation and json serialization)
    # we override BaseModel __iter__ implementation
    # up to now all the behaviors seems working
    @abstractmethod
    def __iter__(self) -> Iterator[ChemDataEntry]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    @abstractmethod
    def get_row(self, idx: int) -> ChemDataEntry: ...

    ###############################################################################################
    #########################                 core methods                #########################
    ###############################################################################################

    def properties_md_table(self, props: Optional[List[PropertyInfo]] = None) -> str:
        """Return Markdown table with name, unit, shape, and description for available properties."""
        props_vals = props or self.available_properties

        def _shape_to_str(shape):
            if isinstance(shape, tuple):
                return "(" + ",".join(str(x) for x in shape) + ")"
            if shape is None:
                return ""
            return str(shape)

        columns = ("name", "unit", "shape", "description")
        rows = [
            (
                p.name,
                "" if p.unit is None else str(p.unit),
                _shape_to_str(p.shape),
                "" if p.description is None else p.description,
            )
            for p in props_vals
        ]
        widths = [max(len(columns[i]), max((len(r[i]) for r in rows), default=0)) for i in range(len(columns))]

        def _fmt_row(row):
            return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

        header = _fmt_row(columns)
        separator = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
        if not rows:
            return "\n".join([header, separator])
        return "\n".join([header, separator, *(_fmt_row(r) for r in rows)])

    @overload
    def __getitem__(self, idx: int) -> ChemDataEntry: ...
    @overload
    def __getitem__(self: TBaseChemDataSet, idx: Union[slice, Iterable[int]]) -> TBaseChemDataSet: ...
    def __getitem__(
        self: TBaseChemDataSet, idx: Union[int, Iterable[int], slice]
    ) -> Union[ChemDataEntry, TBaseChemDataSet]:
        if isinstance(idx, int):
            return self.get_row(idx)
        return self.subset(idx)

    def __len__(self) -> int:
        if self.absolute_idxs is not None:
            return len(self.absolute_idxs)
        return self.total_len()

    @property
    def out_properties(self) -> List[PropertyInfo]:
        return self.composed_transform.properties_out(self.available_properties)

    @property
    def composed_transform(self):
        return ComposedTransform(transforms=self.transforms)

    def subset(
        self: TBaseChemDataSet,
        indices: Optional[Union[int, Iterable[int], slice]] = None,
        relative_idxs: bool = True,
        transfer_transforms: bool = True,
    ) -> TBaseChemDataSet:
        # Delicate: if copy deep then in memory implementations will copy all the data every time.
        # For in memory datasets this might be not that great therefore deep=False!
        # This implies that when __iter__ method is called better to return a deepcopy of the entry!
        # transforms and absolute_idx should be deepcopy
        ds = self.model_copy(deep=False)
        ds.transforms = copy.deepcopy(self.transforms)
        if self.absolute_idxs is None and indices is None:
            # this special case when subset returns a copy of the model.
            ds.absolute_idxs = None
        else:
            ds.absolute_idxs = self.select_indices(indices, relative_idxs=relative_idxs)
        if transfer_transforms:
            ds.transforms = self.transforms.copy()
        return ds

    def restore_original(
        self: TBaseChemDataSet,
    ) -> TBaseChemDataSet:
        """
        Return the original dataset without any selected idxs
        """
        ds = self.subset()
        ds.absolute_idxs = None
        return ds

    ###############################################################################################
    ########################                 extra methods                #########################
    ###############################################################################################
    def from_absolute_to_relative(self, absolute_idxs: np.ndarray):
        absolute = self.select_indices()
        relative_idxs = np.sort(np.searchsorted(absolute, absolute_idxs))
        relative_dtype = np.dtype(np.int64, metadata={"relative_idxs": True})
        return relative_idxs.astype(relative_dtype, copy=False)

    def select_indices(self, indices: Optional[Union[int, Iterable[int], slice]] = None, relative_idxs: bool = True):
        """
        return absolute indices sorted!

        indices from user prospective are always from 0 to N (with N <= len(self)).
        here they are converted in absolute indexes from 0 to len(self), the one stored in absolute_idxs
        then they are converted to the real indexes: like key-value paris, 1-based, ect...

        :param indices: these are indexes relative to the Iterator! not the absolute indexes!, defaults to None
        :type indices: _type_, optional
        :return: _description_
        :rtype: _type_
        """

        # convert indices to np.array
        if indices is None:
            indices = np.arange(len(self), dtype=np.int64)
            relative_idxs = True
        elif isinstance(indices, int):
            indices = np.array([indices], dtype=np.int64)
        elif isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            indices = np.arange(start, stop, step, dtype=np.int64)
            relative_idxs = True
        elif isinstance(indices, np.ndarray):
            if indices.shape == ():
                indices = np.array([indices], dtype=np.int64)
            elif not np.issubdtype(indices.dtype, np.bool_):
                indices = indices.astype(np.int64, copy=False)
        else:
            indices = np.asarray(indices, dtype=np.int64)

        relative_idxs = self._parse_relative_idxs(indices, relative_idxs)

        # intersection operation if relative or absolute
        if self.absolute_idxs is not None:
            if relative_idxs:
                ret = self.absolute_idxs[indices]
            else:
                ret = np.intersect1d(self.absolute_idxs, indices)
        else:
            ret = indices
        # this solve a long standing bug which it is associated to some filtering approaches, as:
        # for i in range(20):
        #     if i in set([1,2,3]):
        #         continue
        # this filters but always iterate in a sorted way, therefore we sort such that any iter_row return sorted data
        # the absolute idx and origin idx are stored, they help inspection
        ret = self.order_and_absolute_type(ret)
        return ret

    def order_and_absolute_type(self, ret: np.ndarray):
        ret.sort()
        ret = ret.astype(self._dtype_absolute_idx(), copy=False)
        return ret

    @classmethod
    def _dtype_absolute_idx(cls):
        return np.dtype(np.int64, metadata={"relative_idxs": False})

    def _parse_relative_idxs(self, absolute_idxs, relative_idxs: bool):
        """if absolute_idxs contains in the metadata that the idxs are relative then switch the flag!"""
        if isinstance(absolute_idxs, np.ndarray):
            md = getattr(getattr(absolute_idxs, "dtype", None), "metadata", {})
            ret = None
            if isinstance(md, dict):
                ret = md.get("relative_idxs", None)
            if ret is not None:
                # print(f"relative_idxs changed because of metadata: From {relative_idxs} to {ret}")
                relative_idxs = ret
        return relative_idxs


# def get_ase_transforms():
#     prop_names_convert = {
#         "dipole_moment": "dipole",
#         "Energy": "energy",
#         "EngineEnergy": "energy",
#         "Gradients": "gradients",
#         "EngineGradients": "gradients",
#     }
#     # self.load_properties = list(set(list(prop_names_convert.keys()) + list(prop_names_convert.values())))
#     transforms = [
#         ConvertNamesTransform(prop_names_convert=prop_names_convert),
#         ToArrayTransform(prop_names=["gradients"], array_type=np.array),
#         ToArrayTransform(prop_names=["energy"], array_type=float),
#         ScalePropTransform(key_in="gradients", key_out="forces", coeff_scale=-1),
#         ReshapeTransform(props_shapes={"forces": (-1, 3), "dipole": (3,)}),
#     ]  # type: ignore
#     return transforms
