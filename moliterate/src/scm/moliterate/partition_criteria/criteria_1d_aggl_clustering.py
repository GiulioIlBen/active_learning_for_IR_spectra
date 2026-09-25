import warnings
from dataclasses import dataclass, fields
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, PrivateAttr
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from typing_extensions import Annotated

from scm.moliterate.core import BaseChemDataSet, BasePartitionCriteria

__all__ = ["AgglomerativeHierarchicalClustering"]


class AgglomerativeHierarchicalClusteringWarning(Warning):
    pass


class AgglomerativeHierarchicalClustering(BasePartitionCriteria):
    type: Literal["AgglomerativeHierarchicalClustering"] = "AgglomerativeHierarchicalClustering"
    # union first so dicts (e.g. YAML) deserialize; any BasePartitionCriteria instance is still accepted from Python
    partition_criterion: Annotated[
        Union["UnionPartitionCriterion", BasePartitionCriteria], Field(union_mode="left_to_right")
    ]
    pdist_metric: Literal[
        "braycurtis",
        "canberra",
        "chebyshev",
        "cityblock",
        "correlation",
        "cosine",
        "dice",
        "euclidean",
        "hamming",
        "jaccard",
        "jensenshannon",
        "kulczynski1",
        "mahalanobis",
        "matching",
        "minkowski",
        "rogerstanimoto",
        "russellrao",
        "seuclidean",
        "sokalmichener",
        "sokalsneath",
        "sqeuclidean",
        "yule",
        "@kabsch_rmsd",
    ] = "euclidean"

    # Active settings only if pdist_metric=="@kabsch_rmsd"
    kabsch_align: Literal["each", "to_one", None] = "each"
    """NOTE: to_one speeds up the alignment but the reference coords to align with, introduces some biases"""
    kabsch_choose_ref: Optional[int] = 0

    # basic explanation: https://www.youtube.com/watch?v=8QCBl-xdeZI&ab_channel=DATAtab
    # the linkage_method determines how to evaluate the distance between clusters
    # complete uses the farthest point sampling approach
    linkage_method: Literal["centroid", "average", "median", "complete", "single"] = "complete"
    # if fcluster_criterion == "distance" then the fcluster_threshold will be of the measure of the distance matrix
    # if fcluster_criterion == "maxclust" then the fcluster_threshold will be of the number of clusters
    # Distance and maxclust need you to decide a height or number in advance.
    # 'inconsistent' instead asks:
    # “keep merging as long as each step looks statistically similar to the ones just below it”.
    fcluster_threshold: Union[float, int] = 0.3
    fcluster_criterion: Literal["distance", "maxclust", "inconsistent", "monocrit"] = "distance"

    _kabsch_chosen_ref: Optional[int] = PrivateAttr(default=None)

    def __call__(self, db: BaseChemDataSet) -> np.ndarray:
        if len(db) < 2:
            return np.arange(len(db))
        dendogram = self.run(db)
        return dendogram.get_results()

    def run(self, db: BaseChemDataSet):
        """this method return a verbose result, better for inspection"""
        assert len(db) > 1
        partition_values = self.partition_criterion(db)
        condensed = self._get_distance_matrix(partition_values)
        dendogram = linkage(condensed, method=self.linkage_method)
        cluster_labels = fcluster(dendogram, t=self.fcluster_threshold, criterion=self.fcluster_criterion)
        t_dist = self.fcluster_threshold
        if self.fcluster_criterion == "inconsistent":
            # t_dist = dist_threshold_from_inconsistent(dendogram, t_inc=self.fcluster_threshold, depth=2)
            t_dist = None
        dendogram_ret = Dendogram(
            cluster_labels, condensed, dendogram, t_dist, self.fcluster_criterion, self.fcluster_threshold
        )
        return dendogram_ret

    def _get_distance_matrix(self, partition_criteria: np.ndarray):
        if self.pdist_metric == "@kabsch_rmsd":
            if self.kabsch_align != "each":
                warnings.warn(
                    (
                        f"Use @kabsch_rmsd with {self.kabsch_align=} only if you know what you are doing. "
                        "With alignment if you choose one system of reference you gain in speed but "
                        "making the alignment dependent on your reference, "
                        "therefore fcluster_threshold or others parameters become reference system dependent"
                    ),
                    AgglomerativeHierarchicalClusteringWarning,
                )
            m, self._kabsch_chosen_ref = compute_rmsd_matrix(
                partition_criteria, align=self.kabsch_align, choose_ref=self.kabsch_choose_ref
            )
            return squareform(
                m,
                force="tovector",
            )
        criteria = np.asarray(partition_criteria, dtype=float).reshape(len(partition_criteria), -1)
        condensed = pdist(criteria, metric=self.pdist_metric)  # type: ignore
        return condensed


@dataclass
class Dendogram:
    cluster_labels: np.ndarray
    condensed_distance_matrix: np.ndarray
    dendrogram: np.ndarray
    threshold: Optional[float]
    criterion: str
    actual_t: float

    @property
    def distance_matrix_NxN(self):
        return squareform(self.condensed_distance_matrix, checks=True, force="tomatrix")

    def get_results(self):
        return self.cluster_labels

    def plot(self, ax=None, **dendrogram_kwargs):
        # Plot the dendrogram
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        dendrogram(self.dendrogram, ax=ax, color_threshold=self.threshold, **dendrogram_kwargs)
        xmin, xmax = ax.get_xlim()
        if self.criterion == "inconsistent":
            pass
        elif self.criterion != "maxclust" and self.threshold is not None:
            ax.hlines(y=self.threshold, xmin=xmin, xmax=xmax, linestyles="dashed")

        ax.set_title(f"Dendrogram C:{self.criterion}, t={self.actual_t}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Distance")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        return ax

    def plot_matrix(self, ax=None):
        # Plot the distance matrix
        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        im = ax.imshow(self.distance_matrix_NxN)  # keep a handle to the image
        ax.set_title("distance_matrix_NxN")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")

        fig.colorbar(im, ax=ax)  # attach colorbar to this image
        return ax

    def __repr__(self) -> str:
        parts = []
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                value_repr = f"np.ndarray(shape={value.shape}, dtype={value.dtype})"
            else:
                value_repr = repr(value)
            parts.append(f"{field.name}={value_repr}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


def get_rmsd(P, Q):
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid
    return np.linalg.norm(Q_centered - P_centered) * np.sqrt(1 / len(P))


def kabsch(P, Q, rotmat=False):
    """
    Align Q to P using the Kabsch algorithm.
    `scipy.spatial.transform.Rotation.align_vectors`:
        52.8 µs ± 429 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    this method:
        29.7 µs ± 222 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    """
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    H = Q_centered.T @ P_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1, 1, d]) @ Vt
    Q_rotated = Q_centered @ R
    Q_ok = Q_rotated + P_centroid
    return (Q_ok, R) if rotmat else Q_ok


def choose_medoid(xyz_s):
    """Return index of structure with minimal total RMSD to all others (brute-force medoid)."""
    mean_centroid = np.mean(np.abs(np.linalg.norm(xyz_s - xyz_s.mean(axis=1)[:, None, :], axis=2)), axis=1)
    return int(np.argmin(mean_centroid))


def compute_rmsd_matrix(
    structures: NDArray, align: Literal["each", "to_one", None] = "to_one", choose_ref: Optional[int] = 0
):
    """
    Compute RMSD similarity matrix with optional alignment.
    structures shape: (n, N_atoms, 3)

    %timeit -n 1000 compute_similarity_matrix(rer_xyz, align="to_one", choose_ref=0) # -> 10 times faster
    %timeit -n 1000 compute_similarity_matrix(rer_xyz, align="each")
        556 µs ± 2.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        6.1 ms ± 54 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    align="each", -> reference and lowest rmsd
    align=None,   -> baseline highest -> 0.04 higher
    align="to_one", choose_ref=0 -> 0.0004 higher than
    """
    n = len(structures)
    ref_idx = None
    if align == "each":
        ref_idx = -1
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                rmsd = get_rmsd(structures[i], kabsch(structures[i], structures[j]))
                matrix[i, j] = matrix[j, i] = rmsd
        return matrix, ref_idx
    elif align == "to_one":
        ref_idx = choose_ref
        if choose_ref is None:
            ref_idx = choose_medoid(structures)
        ref = structures[ref_idx]
        # shape: (n, N_atoms, 3)
        X = np.stack([kabsch(ref, s) if i != ref_idx else s for i, s in enumerate(structures)])
    else:
        X = structures
    # Stack to array of shape (n, N_atoms, 3)
    # Vectorized pairwise RMSD
    diffs = X[:, None, :, :] - X[None, :, :, :]  # shape: (n, n, N_atoms, 3)
    sq_diffs = np.sum(diffs**2, axis=-1)  # shape: (n, n, N_atoms)
    mean_sq = np.mean(sq_diffs, axis=-1)  # shape: (n, n)
    matrix = np.sqrt(mean_sq)  # shape: (n, n)
    return matrix, ref_idx
