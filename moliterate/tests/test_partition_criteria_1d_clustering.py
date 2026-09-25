import numpy as np
import pytest
from pydantic import ConfigDict

from scm.moliterate.core import BaseChemDataSet, BasePartitionCriteria
from scm.moliterate.partition_criteria import AgglomerativeHierarchicalClustering
from scm.moliterate.partition_criteria.criteria_1d_aggl_clustering import AgglomerativeHierarchicalClusteringWarning


class DummyMolData(BaseChemDataSet):
    size: int
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def total_len(self) -> int:
        return self.size

    @property
    def available_properties(self):
        return []

    @property
    def metadata(self):
        return {}

    def __iter__(self):
        return iter(())

    def get_row(self, idx: int):
        raise IndexError(idx)


class StaticValuesCriterion(BasePartitionCriteria):
    values: list
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, db: BaseChemDataSet) -> np.ndarray:
        assert len(db) == len(self.values)
        return np.asarray(self.values, dtype=float)


def test_single_entry_returns_identity_labels():
    db = DummyMolData(size=1)
    criterion = StaticValuesCriterion(values=[0.0])
    clustering = AgglomerativeHierarchicalClustering(partition_criterion=criterion)

    labels = clustering(db)

    np.testing.assert_array_equal(labels, np.array([0]))


def test_numeric_values_cluster_with_distance_threshold():
    values = [0.0, 0.1, 5.0, 5.2]
    db = DummyMolData(size=len(values))
    criterion = StaticValuesCriterion(values=values)
    clustering = AgglomerativeHierarchicalClustering(
        partition_criterion=criterion,
        linkage_method="complete",
        fcluster_threshold=0.5,
        fcluster_criterion="distance",
    )

    labels = clustering(db).ravel()

    assert labels.shape == (4,)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_kabsch_metric_aligns_rotated_structures():
    structures = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],  # rotated version of the first
        [[10.0, 1.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]],
    ]
    db = DummyMolData(size=len(structures))
    criterion = StaticValuesCriterion(values=structures)
    clustering = AgglomerativeHierarchicalClustering(
        partition_criterion=criterion,
        pdist_metric="@kabsch_rmsd",
        fcluster_threshold=0.4,
        linkage_method="complete",
    )

    dendrogram = clustering.run(db)

    # ax = dendrogram.plot()
    # ax.figure.savefig("fig.png")

    assert dendrogram.distance_matrix_NxN.shape == (3, 3)
    assert dendrogram.distance_matrix_NxN[0, 1] == pytest.approx(0.0, abs=1e-6)
    labels = dendrogram.get_results().ravel()
    print("")
    print(dendrogram.condensed_distance_matrix)
    print(labels)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_kabsch_align_each_groups_rotated_structures():
    structures = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        [[10.0, 1.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]],
    ]
    db = DummyMolData(size=len(structures))
    criterion = StaticValuesCriterion(values=structures)
    clustering = AgglomerativeHierarchicalClustering(
        partition_criterion=criterion,
        pdist_metric="@kabsch_rmsd",
        kabsch_align="each",
        fcluster_threshold=0.4,
        linkage_method="complete",
    )

    dendrogram = clustering.run(db)
    assert dendrogram.distance_matrix_NxN[0, 1] == pytest.approx(0.0, abs=1e-6)
    labels = dendrogram.get_results().ravel()
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


@pytest.mark.parametrize("choose_ref", [0, None])
def test_kabsch_choose_ref_variants(choose_ref):
    structures = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # same as next if same atoms
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],  # same as previous if same atoms
        [[10.0, 1.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]],  # different
    ]
    db = DummyMolData(size=len(structures))
    criterion = StaticValuesCriterion(values=structures)
    clustering = AgglomerativeHierarchicalClustering(
        partition_criterion=criterion,
        pdist_metric="@kabsch_rmsd",
        kabsch_align="to_one",
        kabsch_choose_ref=choose_ref,
        fcluster_threshold=0.4,
        linkage_method="complete",
    )
    with pytest.warns(AgglomerativeHierarchicalClusteringWarning):
        dendrogram = clustering.run(db)
    res = clustering.group_by(dendrogram.get_results())
    print("\n", f"{clustering.kabsch_align=} | {choose_ref=} | {clustering._kabsch_chosen_ref=}")
    print(res)
    tri = dendrogram.distance_matrix_NxN[np.triu_indices(3, k=1)]
    print(tri)
    # 2 equals must be clustered in the same cluster and one different
    if choose_ref == 0:
        # Right behavior
        assert set([2, 1]) == set([len(v) for v in res.values()])
        assert set([0, 1]) == set([v for v in res.values() if len(v) == 2][0])
    else:
        # Wrong behavior
        assert set([1, 2, 0]) == set([x[0] for x in res.values()]), "BAD behavior should be deprecated or changed!"


@pytest.mark.parametrize("choose_ref", [0, None])
def test_kabsch_choose_ref_variants_2(choose_ref):
    structures = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # same as next if same atoms
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],  # same as previous if same atoms
        [[10.0, 1.0, 0.0], [10.0, 1.0, 0.0], [10.0, 0.0, 1.0]],  # different
    ]
    db = DummyMolData(size=len(structures))
    criterion = StaticValuesCriterion(values=structures)
    clustering = AgglomerativeHierarchicalClustering(
        partition_criterion=criterion,
        pdist_metric="@kabsch_rmsd",
        kabsch_align="to_one",
        kabsch_choose_ref=choose_ref,
        fcluster_threshold=0.5,
        linkage_method="complete",
    )
    with pytest.warns(AgglomerativeHierarchicalClusteringWarning):
        dendrogram = clustering.run(db)
    res = clustering.group_by(dendrogram.get_results())
    print("\n", f"{clustering.kabsch_align=} | {choose_ref=} | {clustering._kabsch_chosen_ref=}")
    print(res)
    tri = dendrogram.distance_matrix_NxN[np.triu_indices(3, k=1)]
    print(tri)
    if choose_ref == 0:
        # Right behavior, since threshold is too gross
        assert set([3]) == set([len(v) for v in res.values()])
    else:
        # Ultra Wrong behavior: 0 and 2 are different and 0,1 are the same one for permutation symmetry.
        assert set([2, 1]) == set([len(v) for v in res.values()])
        assert set([0, 2]) == set([v for v in res.values() if len(v) == 2][0])


def test_dendrogram_plot_methods():
    values = [0.0, 0.1, 5.0, 5.2]
    db = DummyMolData(size=len(values))
    criterion = StaticValuesCriterion(values=values)
    clustering = AgglomerativeHierarchicalClustering(
        partition_criterion=criterion,
        linkage_method="complete",
        fcluster_threshold=0.5,
        fcluster_criterion="distance",
    )

    dendrogram = clustering.run(db)

    ax = dendrogram.plot()
    ax_matrix = dendrogram.plot_matrix()
    ax.figure.canvas.draw()
    ax_matrix.figure.canvas.draw()
    ax.figure.clf()
    ax_matrix.figure.clf()
