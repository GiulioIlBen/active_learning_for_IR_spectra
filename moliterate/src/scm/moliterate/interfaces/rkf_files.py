"""
Interface for reading atomistic data from RKF-based trajectory files.
"""

import warnings
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np
from pydantic import PrivateAttr, field_validator, model_validator

from scm.moliterate.core.base_chem_dataset import BaseChemDataSet
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo

if TYPE_CHECKING:
    from scm.akfreader import AKFReader
    from scm.plams import KFFile

__all__ = [
    "RKFMolData",
]


class RKFDataError(Exception):
    pass


class RKFDataWarning(Warning):
    pass


_PLAMS_IMPORTS: Optional[Tuple[Any, Any, Any]] = None


def _load_plams_dependencies() -> Tuple[Any, Any, Any]:
    global _PLAMS_IMPORTS
    if _PLAMS_IMPORTS is None:
        try:
            from scm.plams import KFFile, KFHistory, Trajectory
        except ImportError as exc:
            raise RKFDataError("scm.plams is required for RKF file support") from exc
        _PLAMS_IMPORTS = (KFFile, KFHistory, Trajectory)
    return _PLAMS_IMPORTS


class RKFMolData(BaseChemDataSet):
    """RKF database interface for atomistic data."""

    type: Literal["RKFMolData"] = "RKFMolData"
    data_source: str
    sections: Tuple[str, ...] = ("History", "MDHistory")
    exclude_standard_items: Tuple[str, ...] = (
        "Coords",
        "nLatticeVectors",
        "LatticeVectors",
        "Bonds.Index",
        "Bonds.Atoms",
        "Bonds.Orders",
        "Mols.Type",
        "Mols.Atoms",
        "Mols.Index",
    )

    _metadata_cache: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _properties_cache: Optional[List[PropertyInfo]] = PrivateAttr(default=None)
    _sections_map: Dict[str, str] = PrivateAttr(default_factory=dict)
    _akf_reader: Optional["AKFReader"] = PrivateAttr(default=None)

    @property
    def _all_sections(self):
        return list(self._kf_skeleton.keys())

    @property
    def _rk_file(self):
        kf_file, _, _ = _load_plams_dependencies()
        return kf_file(self.data_source)

    @cached_property
    def _kf_skeleton(self):
        return self._rk_file.get_skeleton()

    @property
    def _trajectory(self):
        _, _, trajectory = _load_plams_dependencies()
        return trajectory(self.data_source)

    @field_validator("data_source", mode="before")
    @classmethod
    def _data_source_is_present(cls, value):
        if not Path(value).is_file():
            raise RKFDataError(f"{value} is not a file.")
        return value

    @model_validator(mode="after")
    def _sections_are_correctly_defined(self):
        if "MDHistory" not in self._all_sections and "MDHistory" in self.sections:
            new_sects = list(self.sections)
            new_sects.pop(new_sects.index("MDHistory"))
            self.sections = tuple(new_sects)
            warnings.warn(f"MDHistory is not found. It has been removed {self.sections=}", RKFDataWarning)

        missing_sections = [s for s in self.sections if s not in self._all_sections]

        if missing_sections:
            raise RKFDataError(
                f"Missing sections {missing_sections} in {self._all_sections}"
                f", maybe calculation failed? \n{self.data_source=}"
            )

        if "History" not in self._all_sections:
            raise RKFDataError("History section not found; required to read geometries")

        if "BinLog" in self.sections:
            new_sects = list(self.sections)
            new_sects.pop(new_sects.index("BinLog"))
            self.sections = tuple(new_sects)
            warnings.warn(f"BinLog is not supported yet. It has been removed {self.sections=}", RKFDataWarning)
        return self

    # --------------------------------------------------------------------- #
    # BaseMolIterate interface
    # --------------------------------------------------------------------- #
    def total_len(self) -> int:
        return self._history_length

    @cached_property
    def available_sections(self) -> Tuple[str, ...]:
        return tuple(
            section
            for section, keys in self._kf_skeleton.items()
            if any("nEntries" in key for key in keys) and "EngineResults" != section
        )

    @property
    def available_properties(self) -> List[PropertyInfo]:
        if self._properties_cache is not None:
            return list(self._properties_cache)

        properties: List[PropertyInfo] = []
        sections_map: Dict[str, str] = {}

        for section in self.sections:
            if section not in self._kf_skeleton:
                warnings.warn(f"Section '{section}' not present in '{self.data_source}', skipping.", RKFDataWarning)
                continue

            for name in get_items_section(self._rk_file, section):
                if name in self.exclude_standard_items:
                    continue
                prop_info = get_items_units_and_shapes(self._rk_file, section, [name])[0]
                prop_info = self._add_info_from_akf(section, prop_info)
                prop_info = prop_info.model_copy(
                    update={
                        "name": f"{section}%{prop_info.name}",
                    }
                )
                properties.append(prop_info)
                sections_map[prop_info.name] = section

        self._properties_cache = properties
        self._sections_map = sections_map
        return list(properties)

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata_cache is not None:
            return dict(self._metadata_cache)
        try:
            general_section = self._rk_file.read_section("General")
        except Exception:
            general_section = {}
        metadata = {
            "General": general_section,
            "sections": list(self.sections),
            "data_source": self.data_source,
            "n_entries": self.total_len(),
        }
        self._metadata_cache = metadata
        return dict(metadata)

    def __iter__(self) -> Iterator[ChemDataEntry]:
        transform = self.composed_transform
        absolute_indices = self.select_indices()
        if len(absolute_indices) == 0:
            return
        prop_names = [p.name for p in self.available_properties]
        self._ensure_sections_map()

        _, kf_history, _ = _load_plams_dependencies()
        iterators = [iter(self._trajectory)]
        for name in prop_names:
            iterators.append(kf_history(self._rk_file, self._sections_map[name]).iter(name.split("%")[1]))

        for idx_absolute, values in enumerate(zip(*iterators), start=0):
            if idx_absolute not in absolute_indices:
                continue
            molecule = values[0]
            property_values = values[1:]
            row = ChemDataEntry(
                system=molecule,
                properties={k: v for k, v in zip(prop_names, property_values)},
                idx_absolute=int(idx_absolute),
                idx_origin=int(idx_absolute) + 1,
            )
            yield transform(row)

    def get_row(self, idx: int) -> ChemDataEntry:
        absolute_idx = int(self.select_indices(indices=idx)[0])
        return self._row_by_absolute_idx(absolute_idx)

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #
    @cached_property
    def _history_length(self) -> int:
        return int(self._rk_file.read("History", "nEntries"))

    def _row_by_absolute_idx(self, absolute_idx: int) -> ChemDataEntry:
        self._ensure_sections_map()
        properties: Dict[str, Any] = {}
        idx_origin = absolute_idx + 1  # RKF indexing is 1-based

        for prop in (p.name for p in self.available_properties):
            section = self._sections_map[prop]
            properties[prop] = self._read_kf(idx_origin, prop.split("%")[1], history_section=section)

        row = ChemDataEntry(
            system=self._trajectory[absolute_idx],
            properties=properties,
            metadata={"data_source": self.data_source},
            idx_absolute=absolute_idx,
            idx_origin=idx_origin,
        )
        return self.composed_transform(row)

    def _ensure_sections_map(self):
        if not self._sections_map:
            _ = self.available_properties

    def _add_info_from_akf(self, section: str, prop_info: PropertyInfo) -> PropertyInfo:
        reader = self._get_akf_reader()
        if reader is None:
            return prop_info
        try:
            data_info = reader.description(f"{section}%{prop_info.name}(1)")
        except ValueError:
            return prop_info

        updates: Dict[str, Any] = {}
        unit = data_info.get("_unit")
        shape = self._normalize_shape(data_info.get("_shape"))
        if unit is not None and prop_info.unit is None:
            updates["unit"] = unit
        if shape is not None and prop_info.shape is None:
            updates["shape"] = shape

        if updates:
            return prop_info.model_copy(update=updates)
        return prop_info

    def _normalize_shape(self, shape: Any):
        if shape is None:
            return None
        if isinstance(shape, (list, tuple)):
            shape_list = list(shape)
            if len(shape_list) == 2 and shape_list[1] == ":":
                shape_list = (-1, shape_list[0])
            return tuple(shape_list)
        return shape

    def _get_akf_reader(self) -> Optional["AKFReader"]:
        if self._akf_reader is not None:
            return self._akf_reader
        try:
            from scm.akfreader import AKFReader

            self._akf_reader = AKFReader(self.data_source)
        except Exception:
            self._akf_reader = None
        return self._akf_reader

    def _read_kf(self, step: int, varname: str, history_section: str):
        if self._values_stored_as_blocks(varname, history_section):
            blocksize = int(self._rk_file.read(history_section, "blockSize"))
            iblock = int(np.ceil(step / blocksize))
            value = self._rk_file.read(history_section, f"{varname}({iblock})")
            try:
                value = value[(step - 1) % blocksize]
            except TypeError:
                pass
        else:
            value = self._rk_file.read(history_section, f"{varname}({step})")
        return value

    def _values_stored_as_blocks(self, varname: str, history_section: str):
        n_entries = self._rk_file.read(history_section, "nEntries")
        key_set = self._kf_skeleton.get(history_section, set())
        if "nBlocks" in key_set and f"{varname}({n_entries})" not in key_set:
            return True
        return False

    @staticmethod
    def is_installed():
        _load_plams_dependencies()


def get_items_section(kf_file: "KFFile", section: str, item_identifier: str = "ItemName") -> List[str]:
    """Get the items that are part of a list from the given section."""
    if section not in kf_file:
        raise KeyError(f"Section '{section}' not present in '{kf_file.path}'")
    item_keys = [kn for kn in kf_file.get_skeleton()[section] if item_identifier in kn]
    items = [str(kf_file.read(section, kn)) for kn in item_keys]
    return items


def get_items_units_and_shapes(kf_file: "KFFile", section: str, items: List[str]) -> List[PropertyInfo]:
    """Collect units and shapes information for the provided items."""
    items_units_and_shapes: List[PropertyInfo] = []
    skeleton = kf_file.get_skeleton()[section]
    for item in items:
        unit = None
        shape = None
        units_key = f"{item}(units)"
        shape_per_atom_key = f"{item}(perAtom)"
        last_dim_key = f"{item}(dim)"
        if units_key in skeleton:
            unit = kf_file.read(section, units_key)
        if last_dim_key in skeleton:
            last_shape = kf_file.read(section, last_dim_key)
            if not isinstance(last_shape, list):
                last_shape = [last_shape]
            if shape_per_atom_key in skeleton and kf_file.read(section, shape_per_atom_key):
                shape = ("nAtoms", *last_shape)
            else:
                shape = (*last_shape,)
        items_units_and_shapes.append(PropertyInfo(name=item, unit=unit, shape=shape))
    return items_units_and_shapes
