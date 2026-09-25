from scm.moliterate.filters import RemoveDuplicates
from scm.moliterate.partition_criteria import (
    AgglomerativeHierarchicalClustering,
    ChemFormulaCriterion,
    GetXYZRepresentation,
    TreePartitionCriterion,
)


def test_remove_duplicates_round_trips_through_a_plain_dict():
    duplicates = RemoveDuplicates.build(
        options="AgglXYZEuclKB", num_samples_by_group=1, seed=1, fcluster_threshold=0.05
    )
    duplicates.clustering_algorithm.partition_criteria[-1].kabsch_align = "to_one"

    reloaded = RemoveDuplicates.model_validate(duplicates.model_dump(mode="json"))

    tree = reloaded.clustering_algorithm
    assert isinstance(tree, TreePartitionCriterion)
    assert isinstance(tree.partition_criteria[0], ChemFormulaCriterion)
    clustering = tree.partition_criteria[1]
    assert isinstance(clustering, AgglomerativeHierarchicalClustering)
    assert isinstance(clustering.partition_criterion, GetXYZRepresentation)
    assert clustering.pdist_metric == "@kabsch_rmsd"
    assert clustering.kabsch_align == "to_one"
    assert clustering.fcluster_threshold == 0.05
    assert reloaded.model_dump(mode="json") == duplicates.model_dump(mode="json")
