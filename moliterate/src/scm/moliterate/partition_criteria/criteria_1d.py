from typing import List, Literal, Optional, Tuple

import numpy as np
from pydantic import field_validator

from scm.moliterate.core import (
    BaseChemDataSet,
    BasePartitionCriteria,
    ChemDataEntry,
)
from scm.moliterate.transforms.atoms_stats import MinMaxAtomsDistance
from scm.moliterate.transforms.composed_transforms import ComposedTransform
from scm.moliterate.transforms.property_info_transform import (
    ConvertNamesTransform,
    ReshapeTransform,
    ToArrayTransform,
)
from scm.moliterate.utils.unit_conversion import conversion_ratio

__all__ = [
    "FmaxCriterion",
    "NumberOfAtomsCriterion",
    "PropertyCriterion",
    "MinDistanceCriterion",
    "MaxDistanceCriterion",
    "GyrationRCriterion",
    "ChemFormulaCriterion",
]


class ChemFormulaCriterion(BasePartitionCriteria):
    type: Literal["ChemFormulaCriterion"] = "ChemFormulaCriterion"
    remove_numbers: bool = False

    def __call__(self, db: BaseChemDataSet):
        formulas = []
        if self.remove_numbers:
            for i, row in enumerate(db):
                formulas.append(self.get_formula(row).translate(str.maketrans("", "", "0123456789")))
        else:
            for i, row in enumerate(db):
                formulas.append(self.get_formula(row))
        return np.array(formulas)

    @staticmethod
    def get_formula(data_row: ChemDataEntry) -> str:
        return data_row.atoms.get_chemical_formula()

    @property
    def name(self):
        return "formula"


class NumberOfAtomsCriterion(BasePartitionCriteria):
    type: Literal["NumberOfAtomsCriterion"] = "NumberOfAtomsCriterion"

    def __call__(self, db: BaseChemDataSet):
        return np.array([len(row.atoms) for row in db])

    @property
    def name(self):
        return "NAtoms"


class MinDistanceCriterion(BasePartitionCriteria):
    type: Literal["MinDistanceCriterion"] = "MinDistanceCriterion"
    min_image_convention: bool = False

    def __call__(self, db: BaseChemDataSet):
        trn = MinMaxAtomsDistance(min_out_key="min_distance", mic=self.min_image_convention)
        values = []
        for row in db:
            values.append(trn(row).properties["min_distance"])
        return np.array(values)

    @property
    def name(self):
        return "min(distance)"


class MaxDistanceCriterion(BasePartitionCriteria):
    type: Literal["MaxDistanceCriterion"] = "MaxDistanceCriterion"
    min_image_convention: bool = False

    def __call__(self, db: BaseChemDataSet):
        trn = MinMaxAtomsDistance(min_out_key="max_distance", mic=self.min_image_convention)
        values = []
        for row in db:
            values.append(trn(row).properties["max_distance"])
        return np.array(values)

    @property
    def name(self):
        return "max(distance)"


class GyrationRCriterion(BasePartitionCriteria):
    type: Literal["GyrationRCriterion"] = "GyrationRCriterion"

    def __call__(self, db: BaseChemDataSet):
        return np.array([float(row.get_gyration_radius()) for row in db])

    @property
    def name(self):
        return "GyrationRad"


class PropertyCriterion(BasePartitionCriteria):
    type: Literal["PropertyCriterion"] = "PropertyCriterion"
    property_key: str = "energy"
    post_process: Optional[Literal["l2_max", "l2", "max"]] = None
    unit_transform: Optional[Tuple[str, str]] = None

    @field_validator("unit_transform")
    @classmethod
    def _check_unit_transform(cls, value):
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError(f"Must be 2 but found len(unit_transform)={len(value)}")
        return tuple(value)

    def __call__(self, db: BaseChemDataSet):
        property_values = []
        for row in db:
            property_values.append(row.properties.get(self.property_key, row.metadata.get(self.property_key)))
        if len(property_values) == 0:
            return np.array([])
        property_values = np.array(property_values)
        if self.post_process == "l2_max":
            value = (property_values**2).sum(2).max(1) ** 0.5
        elif self.post_process == "l2":
            value = (property_values**2).sum(2) ** 0.5
        elif self.post_process == "max":
            value = property_values.max(1)
        elif self.post_process is not None:
            raise ValueError(f"Not allowed value: {self.post_process}")
        else:
            value = property_values
        if self.unit_transform is not None:
            value *= conversion_ratio(*self.unit_transform)
        return value

    @property
    def name(self):
        ret = self.property_key
        if self.post_process == "l2_max":
            return f"max(||{ret}||)"
        elif self.post_process == "l2":
            return f"||{ret}||"
        elif self.post_process == "max":
            return f"max({ret})"
        return ret


class FmaxCriterion(BasePartitionCriteria):
    type: Literal["FmaxCriterion"] = "FmaxCriterion"
    forces_keys: Tuple[str, ...] = ("Gradients", "forces", "EngineGradients")
    prepend_keys: Tuple[str, ...] = ("", "History%", "MDHistory%")

    def __call__(self, db: BaseChemDataSet):
        trn = ComposedTransform(
            transforms=[
                ConvertNamesTransform(
                    prop_names_convert={a + k: "forces" for k in self.forces_keys for a in self.prepend_keys}
                ),
                ToArrayTransform(prop_names=["forces"]),
                ReshapeTransform(props_shapes={"forces": [-1, 3]}),
            ]
        )
        property_values = []
        for row in db:
            value = trn(row).properties.get("forces", None)
            assert value is not None, (
                f"You can not collect data on missing values "
                f"@indx={row.idx_absolute}. prop {self.forces_keys} (e.g. forces)"
            )
            value = (value**2).sum(1).max() ** 0.5
            property_values.append(float(value))
        return np.array(property_values)

    @property
    def name(self):
        return "fmax"


class MetadataCriterion(BasePartitionCriteria):
    type: Literal["MetadataCriterion"] = "MetadataCriterion"
    metadata_keys: List[str]
    concat_symbol: str = "%"

    def __call__(self, db: BaseChemDataSet):
        metadata_values = []
        for row in db:
            mds = [str(row.metadata.get(k)) for k in self.metadata_keys]
            metadata_values.append(self.concat_symbol.join(mds))
        metadata_values = np.asarray(metadata_values)
        return metadata_values

    @property
    def name(self):
        return "md[" + self.concat_symbol.join(self.metadata_keys) + "]"
