import warnings
from typing import Iterable, List, Literal, Optional, Union

import numpy as np
from scipy.spatial import distance

from scm.moliterate.core import BaseChemDataSet, BaseFilter
from scm.moliterate.core.filters import BaseFilterError, BaseFilterWarning
from scm.moliterate.partition_criteria import UnionPartitionCriterion


class TopNFilter(BaseFilter):
    type: Literal["TopNFilter"] = "TopNFilter"  # Field discriminator!
    partition_criterion: UnionPartitionCriterion
    num_samples: Union[float, int] = 0.1

    def apply_filter(self, db: BaseChemDataSet) -> Iterable[int]:
        n_data = self._parse_subset_n_data(self.num_samples, len(db))
        feature_values = np.asarray(self.partition_criterion(db), dtype=float)
        if len(db) > np.prod(feature_values.shape):
            raise BaseFilterError(
                f"You can apply this filter ony for 1D partition_criterion"
                f" but {feature_values.shape=} with {len(db)=} and {self.partition_criterion=}"
            )
        elif len(db) < np.prod(feature_values.shape):
            raise BaseFilterError(
                f"Something is wrong here: {feature_values.shape=} with {len(db)=} and {self.partition_criterion=}"
            )
        else:
            feature_values = feature_values.ravel()
        if n_data >= len(feature_values):
            return np.arange(len(feature_values), dtype=np.int64)
        return self._calculate_indices(feature_values, n_data)

    @staticmethod
    def _calculate_indices(feature_values: np.ndarray, n_data: int) -> np.ndarray:
        candidates = np.argpartition(feature_values, -n_data)[-n_data:]
        order = np.argsort(feature_values[candidates])
        return candidates[order]

    def __hash__(self) -> int:  # Needed only for type hint
        return super().__hash__()


class FarthestPointFilter(BaseFilter):
    type: Literal["FarthestPointFilter"] = "FarthestPointFilter"  # Field discriminator!
    partition_criterion: UnionPartitionCriterion
    num_samples: Union[float, int] = 0.1
    seed: Optional[int] = None

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        feature_values = np.asarray(self.partition_criterion(db), dtype=float)
        points = feature_values.reshape(len(db), -1)
        n_data = self._parse_subset_n_data(self.num_samples, len(db))
        if n_data >= len(points):
            warnings.warn(f"No Filter will occur: {n_data=} > {len(points)=}", BaseFilterWarning)
            return None
        test_indices = self.farthest_point_sampling(points, n_data, self.seed)
        return np.asarray(test_indices, dtype=np.int64)

    def plot_idx(self, idx_selected: List[int], db, ax=None):
        """
        If you see that form one bar of the selected is higher than
        the ref is just because of edge cases and for hist
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        feature_values = self.partition_criterion(db)
        points = np.array(feature_values).astype(float).reshape(-1, 1)
        ax.scatter([x for x in range(len(points))], points, label="Original Points", alpha=0.5)
        ax.scatter(idx_selected, points[idx_selected], color="red", label="Sampled Points (FPS)", marker="x")
        # ax.set_ylabel(self.partition_criterion.property_key)
        ax.set_xlabel("idx")
        ax.legend()
        return ax

    @staticmethod
    def farthest_point_sampling(points: np.ndarray, num_samples: int, seed: Optional[int] = None):
        if len(points) == 0 or num_samples <= 0:
            return []
        rng = np.random.default_rng(seed)
        num_samples = min(num_samples, len(points))
        first = int(rng.integers(len(points)))
        sampled_indices = [first]
        min_distances = distance.cdist(points, points[[first]]).ravel()
        min_distances[first] = -np.inf
        for _ in range(1, num_samples):
            next_index = int(np.argmax(min_distances))
            sampled_indices.append(next_index)
            new_distances = distance.cdist(points, points[[next_index]]).ravel()
            min_distances = np.minimum(min_distances, new_distances)
            min_distances[next_index] = -np.inf
        return sampled_indices

    def __hash__(self) -> int:  # Needed only for type hint
        return super().__hash__()


class RandomFilter(BaseFilter):
    type: Literal["RandomFilter"] = "RandomFilter"  # Field discriminator!
    num_samples: Union[float, int] = 0.1
    seed: Optional[int] = None
    make_warning: bool = True

    @property
    def default_rng(self):
        return np.random.default_rng(self.seed)

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        n_entries = len(db)
        n_data = self._parse_subset_n_data(self.num_samples, len(db))
        if n_data >= n_entries:
            if self.make_warning:
                warnings.warn(f"No Filter will occur: {n_data=} > {n_entries=}", BaseFilterWarning)
            return None
        return self.default_rng.choice(n_entries, size=n_data, replace=False)

    def __hash__(self) -> int:  # Needed only for type hint
        return super().__hash__()


class LinearSteppedFilter(BaseFilter):
    """
    Equivalent to slice(None,None,step) >> slice(None, num_samples)
    """

    type: Literal["LinearSteppedFilter"] = "LinearSteppedFilter"  # Field discriminator!
    step: int = 1
    num_samples: Union[float, int] = 0.1

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        n_entries = len(db)
        n_data = self._parse_subset_n_data(self.num_samples, n_entries)

        if self.step == 0:
            raise BaseFilterError(f"Invalid {self.step=}. step must be non-zero.")
        if n_data >= n_entries:
            warnings.warn(f"No Filter will occur: {n_data=} > {n_entries=}", BaseFilterWarning)
            return None

        if self.step > 0:
            sampled_indices = np.arange(0, n_entries, self.step, dtype=np.int64)
        else:
            sampled_indices = np.arange(n_entries - 1, -1, self.step, dtype=np.int64)

        if sampled_indices.size < n_data:
            warnings.warn(
                f"Requested {n_data=} but only {sampled_indices.size=} can be selected with {self.step=}.",
                BaseFilterWarning,
            )
            return sampled_indices

        return sampled_indices[:n_data]

    def __hash__(self) -> int:  # Needed only for type hint
        return super().__hash__()
