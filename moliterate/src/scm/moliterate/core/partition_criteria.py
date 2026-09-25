from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
from pydantic import BaseModel, ConfigDict

from scm.moliterate.core.base_chem_dataset import BaseChemDataSet, IterDataCoreError


class BasePartitionCriteria(ABC, BaseModel):
    # @property
    # @abstractmethod
    # def property_out(self) -> PropertyInfo: ...

    @abstractmethod
    def __call__(self, db: BaseChemDataSet) -> np.ndarray: ...

    @classmethod
    def group_by(cls, cluster_labels: np.ndarray) -> Dict[str, np.ndarray]:
        if len(cluster_labels.shape) == 2 and cluster_labels.shape[1] == 1:
            cluster_labels = cluster_labels.ravel()
        if len(cluster_labels.shape) != 1:
            raise IterDataCoreError(f"To groupby you should have flatten array but {cluster_labels.shape=}")
        unique_labels, inverse = np.unique(cluster_labels, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse)
        splits = np.cumsum(counts[:-1])
        grouped_indices = np.split(order, splits)
        return {str(label): indices for label, indices in zip(unique_labels, grouped_indices)}

    @classmethod
    def to_array(cls, db: BaseChemDataSet, dict_concat: Dict[str, np.ndarray]):
        dtype_len = max((len(k) for k in dict_concat), default=1)
        labels = np.empty(len(db), dtype=f"U{dtype_len}")
        for k, v in dict_concat.items():
            labels[v] = k
        return labels

    @property
    def name(self) -> str:
        return self.__class__.__name__
