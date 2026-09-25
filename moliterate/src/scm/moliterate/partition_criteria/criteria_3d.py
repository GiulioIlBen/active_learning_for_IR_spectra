from typing import Literal

import numpy as np

from ..core import BaseChemDataSet, BasePartitionCriteria

__all__ = ["GetXYZRepresentation"]


def _extract_atoms(row):
    atoms = getattr(row, "atoms", None)
    if atoms is None:
        atoms = row.system
    return atoms


class GetXYZRepresentation(BasePartitionCriteria):
    """works nice without PBC"""

    type: Literal["GetXYZRepresentation"] = "GetXYZRepresentation"
    wrap: bool = False
    check_atoms_ordering: bool = False

    def __call__(self, db: BaseChemDataSet):
        descriptors = []
        numbers = None
        for row in db:
            atoms = _extract_atoms(row)
            descriptors.append(atoms.get_positions(wrap=self.wrap))
            if self.check_atoms_ordering:
                if numbers is None:
                    numbers = atoms.numbers
                if not all(numbers == atoms.numbers):
                    raise ValueError(f"Assumes same atoms ordering but found: {numbers}, {atoms.numbers}")
        return np.array(descriptors)
