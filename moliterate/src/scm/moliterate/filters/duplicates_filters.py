from typing import TYPE_CHECKING, Dict, Literal

import numpy as np

from scm.moliterate.core import BaseChemDataSet, BaseFilter, BasePartitionCriteria
from scm.moliterate.partition_criteria import UnionPartitionCriterion

if TYPE_CHECKING:
    from scm.moliterate.filters import UnionFilters


class RemoveDuplicates(BaseFilter):
    """
    RemoveDuplicates: consists of 2 steps: clustering->data selection
    """

    type: Literal["RemoveDuplicates"] = "RemoveDuplicates"  # Field discriminator!
    clustering_algorithm: UnionPartitionCriterion
    selection_strategy: "UnionFilters"

    def apply_filter(self, db: BaseChemDataSet):
        clusters_labels = self.clustering_algorithm(db)
        clusters = self.clustering_algorithm.group_by(clusters_labels)
        selected_by: Dict[str, BaseChemDataSet] = {}
        for k, v in clusters.items():
            db_cluster = db.subset(v)
            selected_by[k] = self.selection_strategy(db_cluster)
        absolute_idxs = np.concatenate([x.select_indices() for x in selected_by.values()])
        return db.from_absolute_to_relative(absolute_idxs)

    @classmethod
    def build(
        cls,
        options: Literal["AgglGyFmaxRand", "AgglXYZEucl", "AgglXYZEuclKB", "AgglMBTRCos"] = "AgglGyFmaxRand",
        num_samples_by_group=5,
        seed=None,
        fcluster_threshold=0.05,
    ):
        """provides examples on how to combine the classes"""

        # Due to circular import
        from scm.moliterate.filters import RandomFilter
        from scm.moliterate.partition_criteria import (
            AgglomerativeHierarchicalClustering,
            ChemFormulaCriterion,
            GetXYZRepresentation,
            GyrationRFmaxCriterion,
            MBTRCriterion,
            NumberOfAtomsCriterion,
            TreePartitionCriterion,
        )

        # make_warning to False, since it happens frequently that groups are smaller than the num_samples_by_group
        selection_strategy = RandomFilter(num_samples=num_samples_by_group, seed=seed, make_warning=False)
        if options == "AgglGyFmaxRand":
            return cls(
                clustering_algorithm=TreePartitionCriterion(
                    partition_criteria=[
                        ChemFormulaCriterion(),
                        AgglomerativeHierarchicalClustering(
                            partition_criterion=GyrationRFmaxCriterion(),
                            fcluster_threshold=fcluster_threshold,
                        ),
                    ],
                ),
                selection_strategy=selection_strategy,
            )
        if options == "AgglXYZEuclKB":
            return cls(
                clustering_algorithm=TreePartitionCriterion(
                    partition_criteria=[
                        ChemFormulaCriterion(),
                        AgglomerativeHierarchicalClustering(
                            partition_criterion=GetXYZRepresentation(),
                            pdist_metric="@kabsch_rmsd",
                            fcluster_threshold=fcluster_threshold,
                        ),
                    ],
                ),
                selection_strategy=selection_strategy,
            )
        if options == "AgglXYZEucl":
            return cls(
                clustering_algorithm=TreePartitionCriterion(
                    partition_criteria=[
                        ChemFormulaCriterion(),
                        AgglomerativeHierarchicalClustering(
                            partition_criterion=GetXYZRepresentation(),
                            pdist_metric="euclidean",
                            fcluster_threshold=fcluster_threshold,
                        ),
                    ],
                ),
                selection_strategy=selection_strategy,
            )
        if options == "AgglMBTRCos":
            return cls(
                clustering_algorithm=TreePartitionCriterion(
                    partition_criteria=[
                        NumberOfAtomsCriterion(),
                        AgglomerativeHierarchicalClustering(
                            partition_criterion=MBTRCriterion(),
                            pdist_metric="cosine",
                            fcluster_threshold=fcluster_threshold,
                        ),
                    ],
                ),
                selection_strategy=selection_strategy,
            )

    def __hash__(self) -> int:  # Needed only for type hint
        return super().__hash__()
