from __future__ import annotations

import io
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, Literal, Union

from pydantic import BaseModel
from scm.plams import Molecule

if TYPE_CHECKING:
    from scm.base import ChemicalSystem


class AtomisticContainer(BaseModel):
    system_id: str

    @abstractmethod
    def collect_molecules(
        self,
    ) -> Union[Dict[str, Molecule], Molecule, Dict[str, "ChemicalSystem"], "ChemicalSystem"]: ...


class PLAMSMolecule(AtomisticContainer):
    type: Literal["PLAMSMolecule"] = "PLAMSMolecule"
    molecules_xyz: Dict[str, str]

    def collect_molecules(self) -> Dict[str, Molecule]:
        def read_mol(xyz):
            m = Molecule()
            m.readxyz(io.StringIO(xyz))
            return m

        return {name: read_mol(xyz) for name, xyz in self.molecules_xyz.items()}

    @classmethod
    def from_molecules(
        cls,
        system_id: str,
        systems: Union[Molecule, Dict[str, Molecule], str, Dict[str, str]],
    ):
        def _serialize_molecule(molecule: Union[Molecule, str]) -> str:
            if isinstance(molecule, str):
                return molecule
            if not isinstance(molecule, Molecule):
                raise TypeError(f"{type(molecule)=} is not Molecule")

            lines = [str(len(molecule)), ""]
            for atom in molecule:
                x_coord, y_coord, z_coord = atom.coords
                lines.append(f"{atom.symbol} {x_coord:.16g} {y_coord:.16g} {z_coord:.16g}")
            return "\n".join(lines)

        if isinstance(systems, dict):
            molecules_xyz = {name: _serialize_molecule(mol) for name, mol in systems.items()}
        else:
            molecules_xyz = {"": _serialize_molecule(systems)}
        return cls(system_id=system_id, molecules_xyz=molecules_xyz)


class SCMChemicalSystem(AtomisticContainer):
    type: Literal["SCMChemicalSystem"] = "SCMChemicalSystem"
    chemical_systems: Dict[str, str]

    def collect_molecules(self) -> Dict[str, "ChemicalSystem"]:
        from scm.base import ChemicalSystem

        return {name: ChemicalSystem(value) for name, value in self.chemical_systems.items()}

    @classmethod
    def from_molecules(
        cls,
        system_id: str,
        systems: Union["ChemicalSystem", Dict[str, "ChemicalSystem"], Molecule, Dict[str, Molecule]],
    ):
        from scm.base import ChemicalSystem
        from scm.utils.conversions import plams_molecule_to_chemsys

        def _serialize_chemical_system(chemical_system) -> str:
            if isinstance(chemical_system, str):
                return chemical_system
            if isinstance(chemical_system, Molecule):
                chemical_system = plams_molecule_to_chemsys(chemical_system)
            if not isinstance(chemical_system, ChemicalSystem):
                raise TypeError(f"{type(chemical_system)=} is not ChemicalSystem")
            return str(chemical_system)

        if isinstance(systems, dict):
            chemical_systems = {name: _serialize_chemical_system(cs) for name, cs in systems.items()}
        else:
            chemical_systems = {"": _serialize_chemical_system(systems)}
        return cls(system_id=system_id, chemical_systems=chemical_systems)
