"""ASE-backed implementation of the molecule database interface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, Iterable, List, Literal, Optional, Tuple

import numpy as np
from pydantic import PrivateAttr, field_validator

from scm.moliterate.core.base_chem_dataset_writer import BaseChemDataSetWriter
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.transforms import UnionRowTransform
from scm.moliterate.utils.progress_bar import moliterate_tqdm

if TYPE_CHECKING:
    from ase.db.row import AtomsRow as ASEAtomsRow

__all__ = ["ASEMolData"]


class ASEDataError(Exception):
    pass


_ASE_IMPORTS: Optional[Tuple[Any, Any]] = None


def _load_ase_dependencies() -> Tuple[Any, Any]:
    """Lazily import ASE to avoid hard dependency at module import time."""
    global _ASE_IMPORTS
    if _ASE_IMPORTS is None:
        try:
            from ase.db import connect as ase_connect
            from ase.db.row import AtomsRow as ASEAtomsRow
        except ImportError as exc:
            raise ASEDataError("ASE dependencies are required for ASEMolData") from exc
        _ASE_IMPORTS = (ase_connect, ASEAtomsRow)
    return _ASE_IMPORTS


class ASEMolData(BaseChemDataSetWriter):
    """
    Provides an interface to ASE databases format
    But all the properties are stored in data!
    """

    type: Literal["ASEMolData"] = "ASEMolData"
    data_source: str
    _metadata_cache: Optional[Dict[str, Any]] = PrivateAttr(default=None)

    @field_validator("data_source", mode="before")
    @classmethod
    def _validate_data_source(cls, data_source: Any) -> str:
        data_source_str = str(data_source)
        if not os.path.exists(data_source_str):
            raise ASEDataError(f"ASE DB does not exists at {data_source_str}")
        try:
            connect, _ = cls._ase()
            with connect(data_source, use_lock_file=False) as _:
                pass
        except ValueError as e:
            raise ASEDataError(f"ValueError: {e} | For {data_source_str=}")
        return data_source_str

    @classmethod
    def create(
        cls,
        data_source: str = "ase.db",
        available_properties: Optional[List[PropertyInfo]] = None,
        distance_unit: str = "Ang",
        transforms: Optional[List[UnionRowTransform]] = None,
        **kwargs,
    ) -> "ASEMolData":
        data_source = str(data_source)
        Path(data_source).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(data_source):
            raise ASEDataError(f"Dataset already exists: {data_source}")

        # Create the db and insert available_properties
        serialized_props = [cls._serialize_property_info(p) for p in available_properties or []]
        connect, _ = cls._ase()
        with connect(data_source) as conn:
            conn.metadata = {"__properties__": serialized_props, "__distance_unit__": distance_unit}

        return cls(data_source=data_source, transforms=transforms or [], **kwargs)

    def total_len(self) -> int:
        connect, _ = self._ase()
        with connect(self.data_source, use_lock_file=False) as conn:
            return int(conn.count())

    @property
    def available_properties(self) -> List[PropertyInfo]:
        return self.metadata.get("__properties__", [])

    @property
    def distance_unit(self) -> str:
        return self.metadata.get("__distance_unit__", "Ang")

    @available_properties.setter
    def available_properties(self, val: Optional[List[PropertyInfo]]):
        serialized_props = [self._serialize_property_info(p) for p in val or []]
        self.update_metadata(__properties__=serialized_props)

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata_cache is None:
            connect, _ = self._ase()
            with connect(self.data_source, use_lock_file=False) as conn:
                metadata = dict(conn.metadata or {})
            metadata["__properties__"] = [self._to_property_info(p) for p in metadata.get("__properties__", [])]
            self._metadata_cache = metadata
        return self._metadata_cache.copy()

    def __iter__(self) -> Generator[ChemDataEntry, None, None]:
        trn = self.composed_transform
        connect, _ = self._ase()
        with connect(self.data_source, use_lock_file=False) as conn:
            for idx_abs in self.select_indices():
                # TODO there might be a better option to make a query for the sql and then iterate via that
                row = self._get_row_from_conn(conn, int(idx_abs))
                yield trn(row)

    def get_row(self, idx: int) -> ChemDataEntry:
        idxs = [x for x in self.select_indices(indices=idx)]
        connect, _ = self._ase()
        with connect(self.data_source, use_lock_file=False) as conn:
            row = self._get_row_from_conn(conn, int(idxs[0]))
        return self.composed_transform(row)

    def add_systems(self, mol_data_rows: Iterable[ChemDataEntry], **tqdm_settings):
        connect, _ = self._ase()
        with connect(self.data_source) as conn:
            for mol_data_row in moliterate_tqdm(mol_data_rows, **tqdm_settings):
                self._write_row(conn, mol_data_row)

    def add_system(self, mol_data_row: ChemDataEntry):
        connect, _ = self._ase()
        with connect(self.data_source) as conn:
            self._write_row(conn, mol_data_row)

    def update_metadata(self, **kwargs):
        metadata = self.metadata
        metadata.update(kwargs)
        self._write_metadata(metadata)

    def update_row_metadata(self, idx: int, absolute: bool = False, **update):
        idx_absolute = self.select_indices(indices=idx, relative_idxs=(not absolute))[0]
        connect, _ = self._ase()
        with connect(self.data_source) as conn:
            row = conn.get(int(idx_absolute) + 1)
            metadata = self._extract_metadata(row)
            metadata.update(update)
            primitive_meta = {k: v for k, v in metadata.items() if isinstance(v, (str, float, int, bool)) or v is None}
            complex_meta = {k: v for k, v in metadata.items() if k not in primitive_meta}
            if len(complex_meta) == 0:
                conn.update(int(idx_absolute) + 1, **primitive_meta)
            else:
                conn.update(int(idx_absolute) + 1, data={"__metadata__": complex_meta}, **primitive_meta)

    # helpers
    @staticmethod
    def _ase() -> Tuple[Any, Any]:
        return _load_ase_dependencies()

    def _write_row(self, conn, mol_data_row: ChemDataEntry):
        atoms = mol_data_row.atoms
        # VALIDATE properties!
        data = {k.name: self._ase_safe_value(mol_data_row.properties[k.name]) for k in self.available_properties}
        metadata = dict(mol_data_row.metadata or {})
        primitive_meta = {}
        if metadata:
            primitive_meta = {k: v for k, v in metadata.items() if isinstance(v, (str, float, int, bool)) or v is None}
            complex_meta = {k: v for k, v in metadata.items() if k not in primitive_meta}
            if complex_meta:
                data["__metadata__"] = complex_meta
        conn.write(atoms, data=data, **primitive_meta)

    @staticmethod
    def _ase_safe_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _get_row_from_conn(self, conn, idx_abs: int):
        """handle idx and idx_abs to be written correctly"""
        idx_origin = idx_abs + 1
        atoms_row = conn.get(int(idx_origin))
        row = self._atoms_row_to_mol_data_row(atoms_row)
        row.idx_absolute = idx_abs
        row.idx_origin = idx_origin
        return row

    def _atoms_row_to_mol_data_row(self, atoms_row: ASEAtomsRow) -> ChemDataEntry:
        """conversion ASEAtomsRow to MolDataRow"""
        atoms = atoms_row.toatoms(add_additional_information=False)
        properties = dict(atoms_row.data or {})
        if atoms.calc is not None:
            if atoms.calc.results:
                properties.update(atoms.calc.results)
        metadata = self._extract_metadata(atoms_row, data=properties)
        properties.pop("__metadata__", None)
        return ChemDataEntry(system=atoms, properties=properties, metadata=metadata)

    @staticmethod
    def _extract_metadata(atoms_row: ASEAtomsRow, data: Optional[Dict] = None) -> Dict[str, Any]:
        metadata = dict(atoms_row.key_value_pairs)
        if data is None:
            data = atoms_row.data
        extra = data.get("__metadata__", {})
        if isinstance(extra, dict):
            metadata.update(extra)
        return metadata

    @staticmethod
    def _serialize_property_info(prop: Any) -> Dict[str, Any]:
        if isinstance(prop, PropertyInfo):
            return prop.model_dump()
        if isinstance(prop, dict):
            return prop
        raise ASEDataError(f"Invalid PropertyInfo entry: {prop!r}")

    @staticmethod
    def _to_property_info(prop: Any) -> PropertyInfo:
        if isinstance(prop, PropertyInfo):
            return prop
        if isinstance(prop, dict):
            return PropertyInfo(**prop)
        raise ASEDataError(f"Invalid PropertyInfo entry in metadata: {prop!r}")

    def _write_metadata(self, metadata: Dict[str, Any]):
        self._metadata_cache = None
        serialized = dict(metadata)
        serialized["__properties__"] = [self._serialize_property_info(p) for p in metadata.get("__properties__", [])]
        connect, _ = self._ase()
        with connect(self.data_source) as conn:
            conn.metadata = serialized
