from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from scm.moliterate.partition_criteria.criteria_1d import (
    ChemFormulaCriterion,
    FmaxCriterion,
    GyrationRCriterion,
    MaxDistanceCriterion,
    MetadataCriterion,
    MinDistanceCriterion,
    NumberOfAtomsCriterion,
    PropertyCriterion,
)
from scm.moliterate.partition_criteria.criteria_1d_aggl_clustering import (
    AgglomerativeHierarchicalClustering,
)
from scm.moliterate.partition_criteria.criteria_1d_splitting import (
    BinaryRandomSplitCriterion,
    SequentialSplitCriterion,
    StratifiedSplitCriterion,
    TrainValidationTestSplitCriterion,
)
from scm.moliterate.partition_criteria.criteria_2d import (
    GetMACERepresentation,
    GyrationRFmaxCriterion,
    MBTRCriterion,
)
from scm.moliterate.partition_criteria.criteria_3d import GetXYZRepresentation
from scm.moliterate.partition_criteria.mol_connectivity_1d import MolConnectivityCriterion
from scm.moliterate.partition_criteria.tree_1d_criteria import (
    TreePartitionCriterion,
)

UnionPartitionCriterion = Annotated[
    Union[
        AgglomerativeHierarchicalClustering,
        BinaryRandomSplitCriterion,
        ChemFormulaCriterion,
        FmaxCriterion,
        GetMACERepresentation,
        GetXYZRepresentation,
        GyrationRCriterion,
        GyrationRFmaxCriterion,
        MaxDistanceCriterion,
        MBTRCriterion,
        MetadataCriterion,
        MinDistanceCriterion,
        NumberOfAtomsCriterion,
        PropertyCriterion,
        SequentialSplitCriterion,
        StratifiedSplitCriterion,
        TrainValidationTestSplitCriterion,
        MolConnectivityCriterion,
        TreePartitionCriterion,
    ],
    Field(discriminator="type"),
]

# Composite criteria reference the union by name so they can be (de)serialized, e.g. from YAML.
AgglomerativeHierarchicalClustering.model_rebuild(_types_namespace={"UnionPartitionCriterion": UnionPartitionCriterion})
TreePartitionCriterion.model_rebuild(_types_namespace={"UnionPartitionCriterion": UnionPartitionCriterion})

__all__ = [
    "AgglomerativeHierarchicalClustering",
    "ChemFormulaCriterion",
    "FmaxCriterion",
    "NumberOfAtomsCriterion",
    "PropertyCriterion",
    "MinDistanceCriterion",
    "MaxDistanceCriterion",
    "GyrationRCriterion",
    "MolConnectivityCriterion",
    "MetadataCriterion",
    # 2d criteria
    "GyrationRFmaxCriterion",
    "MBTRCriterion",
    "GetMACERepresentation",
    # 3d criteria
    "GetXYZRepresentation",
    #
    "TreePartitionCriterion",
    # splitting
    "BinaryRandomSplitCriterion",
    "TrainValidationTestSplitCriterion",
    "SequentialSplitCriterion",
    "StratifiedSplitCriterion",
    # all
    "UnionPartitionCriterion",
]
