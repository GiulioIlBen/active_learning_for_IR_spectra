from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Literal, Optional, Type, Union, overload

import numpy as np
from pydantic import BaseModel, ConfigDict

from scm.moliterate.core import BaseChemDataSet, BaseFilter
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.partition_criteria.mol_connectivity_1d import MolConnectivityCriterion

if TYPE_CHECKING:
    from scm.plams import Molecule

_IMPL_CACHE_CONF_SCM_CLS = None


def _get_impls() -> Dict[str, Any]:
    global _IMPL_CACHE_CONF_SCM_CLS
    if _IMPL_CACHE_CONF_SCM_CLS is None:
        try:
            from scm.conformers import (
                UniqueConformersAMS,
                UniqueConformersCrest,
                UniqueConformersRMSD,
                UniqueConformersTFD,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("scm.conformers is required to use AMSConformersFilter") from exc
        _IMPL_CACHE_CONF_SCM_CLS = {
            "ams": UniqueConformersAMS,
            "crest": UniqueConformersCrest,
            "rmsd": UniqueConformersRMSD,
            "tfd": UniqueConformersTFD,
        }
    return _IMPL_CACHE_CONF_SCM_CLS


class _BaseConformerSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def build(self) -> Any:
        raise NotImplementedError


class UniqueConformersAMSSettings(_BaseConformerSettings):
    energy_threshold: float = 0.2
    dihedral_threshold: float = 30.0
    distance_threshold: float = 0.1
    # extra
    use_torsions: bool = True
    # H do not exists (to unset this put empty list?)! you can add a list of extra atoms
    irrelevant_atoms: Optional[List[int]] = None
    max_dist: Optional[float] = None
    compare_across_molecules: bool = True

    def build(self) -> Any:
        impls = _get_impls()
        conf = impls["ams"](
            energy_threshold=self.energy_threshold,
            dihedral_threshold=self.dihedral_threshold,
            distance_threshold=self.distance_threshold,
        )
        settings = conf.settings
        settings.energy_threshold = self.energy_threshold
        settings.dihedral_threshold = self.dihedral_threshold
        settings.distance_threshold = self.distance_threshold
        settings.use_torsions = self.use_torsions
        settings.irrelevant_atoms = self.irrelevant_atoms
        settings.max_dist = self.max_dist
        settings.compare_across_molecules = self.compare_across_molecules
        return conf


class UniqueConformersRMSDSettings(_BaseConformerSettings):
    energy_threshold: float = 0.05
    rmsd_threshold: float = 0.125

    def build(self) -> Any:
        impls = _get_impls()
        conf = impls["rmsd"](
            energy_threshold=self.energy_threshold,
            rmsd_threshold=self.rmsd_threshold,
        )
        settings = conf.settings
        settings.energy_threshold = self.energy_threshold
        settings.rmsd_threshold = self.rmsd_threshold
        return conf


class UniqueConformersCrestSettings(_BaseConformerSettings):
    energy_threshold: float = 0.05
    rmsd_threshold: float = 0.125
    bconst_threshold: float = 0.003
    # extra
    bthr_max: float = 0.025
    bthr_shift: float = 0.5
    bconst_scaling_factor: float = 15.0
    size_threshold: float = 1e-3
    follow_paper: bool = False

    def build(self) -> Any:
        impls = _get_impls()
        conf = impls["crest"](
            energy_threshold=self.energy_threshold,
            rmsd_threshold=self.rmsd_threshold,
            bconst_threshold=self.bconst_threshold,
        )
        settings = conf.settings
        settings.energy_threshold = self.energy_threshold
        settings.rmsd_threshold = self.rmsd_threshold
        settings.scaled_rotational_constant_settings.rotational_constant_threshold = self.bconst_threshold
        settings.scaled_rotational_constant_settings.bthr_max = self.bthr_max
        settings.scaled_rotational_constant_settings.bthr_shift = self.bthr_shift
        settings.bconst_scaling_factor = self.bconst_scaling_factor
        settings.size_threshold = self.size_threshold
        settings.follow_paper = self.follow_paper
        return conf


class UniqueConformersTFDSettings(_BaseConformerSettings):
    energy_threshold: float = 0.05
    tfd_threshold: float = 0.05
    # extra
    use_energy_threshold: bool = True
    use_weights: bool = True

    def build(self) -> Any:
        impls = _get_impls()
        conf = impls["tfd"](
            energy_threshold=self.energy_threshold,
            tfd_threshold=self.tfd_threshold,
        )
        settings = conf.settings
        settings.use_energy_threshold = self.use_energy_threshold
        settings.energy_threshold = self.energy_threshold
        settings.tfd_threshold = self.tfd_threshold
        settings.use_weights = self.use_weights
        return conf


@dataclass(frozen=True)
class _RowInfo:
    idx: int
    mol: "Molecule"
    energy: float
    connectivity_hash: Optional[str] = None


class AMSConformersFilter(BaseFilter):
    """
    Filter conformers using AMS conformers implementations while keeping the selection logic in one place.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    type: Literal["AMSConformersFilter"] = "AMSConformersFilter"  # Field discriminator!

    # Grouping options
    grouping_criterion: Optional[MolConnectivityCriterion] = MolConnectivityCriterion()
    reorder_on_hash_connectivity: bool = True

    # filtering settings
    settings: Union[
        UniqueConformersAMSSettings,
        UniqueConformersCrestSettings,
        UniqueConformersRMSDSettings,
        UniqueConformersTFDSettings,
    ] = UniqueConformersAMSSettings()

    # Energy extraction
    energy_property_key: Optional[str] = None
    energy_extractor: Optional[Callable[[ChemDataEntry], float]] = None

    # Pass-through options for add_conformers
    max_energy: Optional[float] = None
    log_data: bool = False

    #########################################################################
    #########################################################################
    #########################################################################

    @classmethod
    @overload
    def settings_class(cls, implementation: Literal["ams"] = "ams") -> Type[UniqueConformersAMSSettings]: ...

    @classmethod
    @overload
    def settings_class(cls, implementation: Literal["crest"]) -> Type[UniqueConformersCrestSettings]: ...

    @classmethod
    @overload
    def settings_class(cls, implementation: Literal["rmsd"]) -> Type[UniqueConformersRMSDSettings]: ...

    @classmethod
    @overload
    def settings_class(cls, implementation: Literal["tfd"]) -> Type[UniqueConformersTFDSettings]: ...

    @classmethod
    def settings_class(
        cls, implementation: Literal["ams", "crest", "rmsd", "tfd"] = "ams"
    ) -> Union[
        Type[UniqueConformersAMSSettings],
        Type[UniqueConformersCrestSettings],
        Type[UniqueConformersRMSDSettings],
        Type[UniqueConformersTFDSettings],
    ]:
        impl = str(implementation).lower()
        mapping = {
            "ams": UniqueConformersAMSSettings,
            "crest": UniqueConformersCrestSettings,
            "rmsd": UniqueConformersRMSDSettings,
            "tfd": UniqueConformersTFDSettings,
        }
        try:
            return mapping[impl]
        except KeyError as exc:
            raise ValueError(f"Unknown implementation: {implementation!r}") from exc

    #########################################################################
    #########################################################################
    #########################################################################

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        if len(db) == 0:
            return None

        groups = self._collect_rows(db)
        selected_indices: List[int] = []

        for grouped_entries in groups.values():
            if not grouped_entries:
                continue
            conf = self.settings.build()
            self._patch_reorder_compat(conf)
            reference_mol = grouped_entries[0].mol

            conf.prepare_state(reference_mol)

            geometries: List[np.ndarray] = []
            energies: List[float] = []
            for grouped_rows in grouped_entries:
                geometries.append(self._reorder_and_get_xyz(grouped_rows.mol, reference_mol))
                energies.append(grouped_rows.energy)

            conf.add_conformers(
                geometries=geometries,
                energies=energies,
                max_energy=self.max_energy,
                log_data=self.log_data,
            )

            candidate_indices = self._candidate_indices_list(conf.candidate_indices)
            selected_indices.extend(grouped_entries[i].idx for i in candidate_indices)

        return np.array(sorted(selected_indices), dtype=np.int64)

    #########################################################################
    #########################################################################
    #########################################################################

    def _collect_rows(self, db: BaseChemDataSet):
        def get_row(i, row, mol, hash_i):
            if hash_i is None:
                MolConnectivityCriterion._raise_if_periodic(row)
            return _RowInfo(
                idx=i,
                mol=mol,
                energy=self._extract_energy(row),
                connectivity_hash=hash_i,
            )

        if self.grouping_criterion is None:
            return {"": [get_row(i, row, row.molecule, None) for i, row in enumerate(db)]}
        groups = {}
        for i, (hash_i, mol, row) in enumerate(self.grouping_criterion._iter_grouping_data(db)):
            groups.setdefault(hash_i, []).append(get_row(i, row, mol, hash_i))
        return groups

    def _reorder_and_get_xyz(self, mol: "Molecule", reference: "Molecule"):
        ret = self._reorder_molecule(mol, reference=reference)
        return self._geometry_from_mol(ret)

    def _reorder_molecule(self, mol: "Molecule", reference: "Molecule") -> "Molecule":
        if not self.reorder_on_hash_connectivity:
            return mol
        reordered = mol.reorder(reference)
        if reordered is None:
            return mol
        return reordered

    @staticmethod
    def _geometry_from_mol(mol: "Molecule") -> np.ndarray:
        return np.asarray(mol.as_array())

    def _extract_energy(self, row: ChemDataEntry) -> float:
        if self.energy_extractor is not None:
            value = self.energy_extractor(row)
            return self._energy_to_float(value)

        if self.energy_property_key:
            value = row.properties.get(self.energy_property_key, row.metadata.get(self.energy_property_key))
            return self._energy_to_float(value)
        return 0.0

    @staticmethod
    def _energy_to_float(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return float(value)
            if value.size == 1:
                return float(value.item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _candidate_indices_list(candidate_indices: Any) -> List[int]:
        if hasattr(candidate_indices, "_values"):
            return list(candidate_indices._values)
        if isinstance(candidate_indices, np.ndarray):
            return candidate_indices.astype(int).tolist()
        try:
            return list(candidate_indices)
        except TypeError:
            return [candidate_indices[i] for i in range(len(candidate_indices))]

    def _patch_reorder_compat(self, conf: Any) -> None:
        if not isinstance(self.settings, UniqueConformersRMSDSettings):
            return
        if getattr(conf, "_moliterate_reorder_patch", False):
            return

        conformer_data = getattr(conf, "conformer_data", None)
        if not isinstance(conformer_data, dict):
            return
        rdconfs = conformer_data.get("trimmed_rdconfs")
        if rdconfs is None or getattr(rdconfs, "_moliterate_reorder_patch", False):
            return

        def reorder_rdconfs(instance: Any, indices: Any) -> None:
            from rdkit import Chem

            copied_conformers = [Chem.Conformer(instance._rdmol.GetConformer(int(i))) for i in indices]
            instance._rdmol.RemoveAllConformers()
            for conformer in copied_conformers:
                instance._rdmol.AddConformer(conformer, assignId=True)

        rdconfs.reorder = MethodType(reorder_rdconfs, rdconfs)
        rdconfs._moliterate_reorder_patch = True
        conf._moliterate_reorder_patch = True

    #########################################################################
    #########################################################################
    #########################################################################

    def hash_payload(self) -> Any:
        return {
            "type": self.type,
            "settings": self.settings.model_dump(mode="json"),
            "grouping_criterion": (
                self.grouping_criterion
                if self.grouping_criterion is None
                else self.grouping_criterion.model_dump(mode="json")
            ),
            "reorder_on_hash_connectivity": self.reorder_on_hash_connectivity,
            "energy_property_key": self.energy_property_key,
            "energy_extractor": self._callable_id(self.energy_extractor),
            "max_energy": self.max_energy,
        }

    @staticmethod
    def _callable_id(value: Optional[Callable[..., Any]]) -> Optional[str]:
        if value is None:
            return None
        return getattr(value, "__qualname__", repr(value))

    def __hash__(self) -> int:  # Needed only for type hint
        return super().__hash__()
