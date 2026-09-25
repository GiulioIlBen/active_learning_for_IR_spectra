from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from pydantic import BaseModel, ConfigDict, JsonValue, field_validator, model_validator

if TYPE_CHECKING:
    from scm.base import ChemicalSystem
    from scm.plams import Molecule


def _import_plams_dependencies():
    try:
        from scm.plams import Molecule
        from scm.plams.interfaces.molecule.ase import (
            fromASE as _ase_to_plams,
        )
        from scm.plams.interfaces.molecule.ase import toASE as _plams_to_ase

        return Molecule, _ase_to_plams, _plams_to_ase
    except Exception as e:
        raise ModuleNotFoundError(f"plams is not installed: pip install plams>=2025.104 \n{e}")


def _import_chem_system_libbase():
    try:
        try:
            from scm.base import ChemicalSystem
        except Exception:
            from scm.libbase import UnifiedChemicalSystem as ChemicalSystem

        return ChemicalSystem
    except Exception:
        raise ModuleNotFoundError("libbase is not installed (use: from scm.base import ChemicalSystem)")


class ChemDataEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    system: Atoms
    properties: Dict[str, Any] = {}
    metadata: Dict[str, JsonValue] = {}
    idx_absolute: int = -1
    idx_origin: Union[int, str] = -1

    @field_validator("system", mode="before")
    @classmethod
    def _ensure_atoms(cls, value):
        if isinstance(value, Atoms):
            return value
        if value.__class__.__name__ == "Molecule":
            _, _, _plams_to_ase = _import_plams_dependencies()
            return _plams_to_ase(value)
        if value.__class__.__name__ == "ChemicalSystem":
            ChemicalSystem = _import_chem_system_libbase()
            return ChemicalSystem.to_ase_atoms(value)
        raise TypeError(f"system must be ase.Atoms; got {type(value)!r}")

    @field_validator("metadata", mode="before")
    @classmethod
    def _validate_metadata(cls, value):
        if value is None:
            return {}
        if not isinstance(value, dict):
            value = dict(value)
        for key, val in value.items():
            if not isinstance(key, str):
                raise TypeError("metadata keys must be strings")
            if not cls._is_json_compatible(val):
                raise TypeError(f"metadata value for key '{key}' is not JSON compatible: {val!r}")
        return value

    @model_validator(mode="after")
    def _ensure_atoms_info_for_chem(self):
        fields_to_pass = ["charge", "ams_region"]
        for k in fields_to_pass:
            if k in self.system.info:
                self.metadata[k] = self.system.info[k]
        return self

    @classmethod
    def _is_json_compatible(cls, value: Any) -> bool:
        if value is None or isinstance(value, (str, int, float, bool)):
            return True
        if isinstance(value, dict):
            return all(isinstance(k, str) and cls._is_json_compatible(v) for k, v in value.items())
        if isinstance(value, (list, tuple)):
            return all(cls._is_json_compatible(v) for v in value)
        return False

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        formula = getattr(self.system, "get_chemical_formula", None)
        if callable(formula):
            formula = formula()
        return (
            f"{self.__class__.__name__}({formula}, prop:{list(self.properties.keys())}, "
            f"md:{list(self.metadata.keys())}, idx_absolute={self.idx_absolute}, idx_origin={self.idx_origin})"
        )

    ###############################################################################################
    ##############################               converters              ##########################
    ###############################################################################################

    @property
    def atoms(self) -> Atoms:
        return self.system

    @property
    def molecule(self) -> "Molecule":
        _, _ase_to_plams, _ = _import_plams_dependencies()
        return _ase_to_plams(self.system)

    @property
    def chemical_system(self) -> "ChemicalSystem":
        ChemicalSystem = _import_chem_system_libbase()
        atoms = self.atoms.copy()
        atoms.info.update(self.metadata)
        return ChemicalSystem.from_ase_atoms(self.system)

    def to_ase_atoms(self, suffix_origin: Optional[str] = "") -> "Atoms":
        """_summary_

        :param suffix_origin: if equal to '_' and add to another dataset you
                              can loose this information because they might be overridden, defaults to "".
                              If None this data will not be added.
        :type suffix_origin: Optional[str], optional
        :return: ase.Atoms for metadata in the .info and properties in .calc.results.
                 Special metadata about the row origin are stored in
                 f"{suffix_origin}origin_idx" f"{suffix_origin}idx" ,f"{suffix_origin}origin_sha"
        :rtype: Atoms
        """
        atoms = self.atoms
        atoms.info.update(self.metadata)
        if suffix_origin is not None:
            atoms.info[f"{suffix_origin}idx_origin"] = self.idx_origin
            atoms.info[f"{suffix_origin}idx_absolute"] = self.idx_absolute
        atoms.calc = SinglePointCalculator(atoms)
        atoms.calc.results.update(self.properties)
        return atoms

    def to_plams_molecule(
        self,
        metadata_in_first_atom_info: bool = True,
        suffix_origin: Optional[str] = "",
        transfer_properties: bool = False,
    ) -> "Molecule":
        """convert to a molecule object that self contains all the metadata

        :param suffix_origin: if equal to '_' and add to another dataset you can loose
                              this information because they might be overridden, defaults to "".
                              If None this data will not be added.
        :type suffix_origin: Optional[str], optional
        :return: ase.Atoms for metadata in the .info and properties in .calc.results.
                Special metadata about the row origin are stored in
                f"{suffix_origin}origin_idx" f"{suffix_origin}idx" ,f"{suffix_origin}origin_sha"
        :rtype: Atoms
        """
        mol = self.molecule
        metadata = self.metadata
        if suffix_origin is not None:
            metadata[f"{suffix_origin}idx_origin"] = self.idx_origin
            metadata[f"{suffix_origin}idx_absolute"] = self.idx_absolute
        if metadata_in_first_atom_info:
            mol[1].properties.info = metadata
        else:
            mol.properties.info = metadata
        if transfer_properties:
            mol.properties.update(self.properties)
        return mol

    ###############################################################################################
    ############################               systems repr              ##########################
    ###############################################################################################

    def get_gyration_radius(self) -> float:
        positions = self.system.get_positions()
        masses = self.system.get_masses()
        total_mass = masses.sum()
        center = (masses[:, None] * positions).sum(axis=0) / total_mass
        diffs = positions - center
        return float(np.sqrt((masses * (diffs * diffs).sum(axis=1)).sum() / total_mass))
