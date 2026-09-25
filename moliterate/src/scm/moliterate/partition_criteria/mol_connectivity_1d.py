from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterable, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import field_validator

from scm.moliterate.core import BaseChemDataSet, BasePartitionCriteria, ChemDataEntry

if TYPE_CHECKING:
    from scm.plams import Molecule


class MolConnectivityCriterion(BasePartitionCriteria):
    type: Literal["MolConnectivityCriterion"] = "MolConnectivityCriterion"
    hash_connectivity_level: Literal[
        "formula",
        "single_bond_connectivity",
        "bond_orders",
        "stereochemistry",
        "rotamers",
    ] = "single_bond_connectivity"
    flags: Optional[Dict[str, Union[bool, float]]] = None  # Advanced settings
    _hash_connectivity_levels: ClassVar[Dict[str, int]] = {
        "formula": 0,
        "single_bond_connectivity": 1,
        "bond_orders": 2,
        "stereochemistry": 3,
        "rotamers": 4,
    }

    @field_validator("hash_connectivity_level", mode="before")
    @classmethod
    def _normalize_hash_connectivity_level(cls, value: Any) -> Any:
        reverse_hash = {level: label for label, level in cls._hash_connectivity_levels.items()}
        if isinstance(value, int):
            try:
                return reverse_hash[value]
            except KeyError as exc:
                raise ValueError(f"Unsupported hash_connectivity_level: {value!r}") from exc
        return value

    @property
    def hash_connectivity_level_value(self) -> int:
        return self._hash_connectivity_levels[self.hash_connectivity_level]

    def __call__(self, db: BaseChemDataSet) -> np.ndarray:
        labels = []
        for conn_hash, _, _ in self._iter_grouping_data(db):
            labels.append(conn_hash)
        return np.asarray(labels)

    def _iter_grouping_data(self, db: BaseChemDataSet) -> Iterable[Tuple[str, "Molecule", ChemDataEntry]]:
        for row in db:
            self._raise_if_periodic(row)
            mol = row.molecule
            # `connectivity_hash` is a string since `hash_connectivity_level_value` is int
            connectivity_hash = mol.label(
                level=self.hash_connectivity_level_value,
                flags=self.flags,
                keep_labels=True,
            )
            yield connectivity_hash, mol, row  # pyright: ignore[reportReturnType]

    @property
    def name(self) -> str:
        vals = [x.capitalize() for x in self.hash_connectivity_level.split("_")]
        camel_case = "".join(vals)
        return f"Mol{camel_case}"

    @staticmethod
    def _raise_if_periodic(row: ChemDataEntry) -> None:
        atoms = row.atoms
        try:
            pbc = atoms.get_pbc()
        except Exception:
            pbc = getattr(atoms, "pbc", None)
        if pbc is None:
            return
        if np.any(pbc):
            raise ValueError("Periodic systems are not supported.")
